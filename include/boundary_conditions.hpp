#pragma once
#include "gas_models.hpp"
#include "macro_and_libc.hpp"
#include "riemann_solver.hpp"

namespace numPDE
{
    enum class BCType
    {
        Wall,       // Reflective (Slip)
        Outflow,    // Transmissive (Zero-gradient)
        Dirichlet,  // Fixed constant state
        TimeInflow, // Time-dependent Dirichlet
        Custom,     // Arbitrary user lambda
        Periodic    // Periodic BC
    };
    enum class Direction
    {
        Left = 0,
        Right,
        Top,
        Bottom
    };

    // ====================================================================
    // C++20 CONCEPT: Factory Lambda Enforcer
    // ====================================================================
    template <typename Func, typename T, std::size_t Dims>
    concept ValidFactoryBC =
        requires(Func f, PetscReal t, const ComprFD::NodeState<T, Dims>& state) {
            { f(t, state) } -> std::convertible_to<ComprFD::NodeState<T, Dims>>;
        };

    // ====================================================================
    // BASE CLASS: Holds all the generic binding logic
    // ====================================================================
    template <std::floating_point T, std::size_t Dims>
    class BoundaryConditionsBase
    {
      public:
        static constexpr std::size_t n_faces = 2 * Dims;

        using State     = ComprFD::NodeState<T, Dims>;
        using CplxState = ComprFD::NodeState<CplxType, Dims>;

        using RealFunc = std::function<State(PetscReal, const State&)>;
        using CplxFunc = std::function<CplxState(PetscReal, const CplxState&)>;

        std::array<BCType, n_faces>   m_types;
        std::array<RealFunc, n_faces> m_bc_real;
        std::array<CplxFunc, n_faces> m_bc_cplx;

        // Default initialize all faces to Outflow
        BoundaryConditionsBase() { m_types.fill(BCType::Outflow); }

        // ====================================================================
        // DRY BINDERS: Centralized logic for all dimensions
        // ====================================================================
        void bind_native_bc(Direction dir, BCType type)
        {
            auto face_idx     = static_cast<std::size_t>(dir);
            m_types[face_idx] = type;

            if (type == BCType::Wall)
            {
                // Wall reflects the momentum normal to the face.
                // X-axis is 0 (Faces 0,1), Y-axis is 1 (Faces 2,3)
                const std::size_t axis = face_idx / 2;

                m_bc_real[face_idx] = [axis](PetscReal, const State& st)
                {
                    State res      = st;
                    res.rhoU[axis] = -res.rhoU[axis]; // Invert normal momentum
                    return res;
                };
                m_bc_cplx[face_idx] = [axis](PetscReal, const CplxState& st)
                {
                    CplxState res  = st;
                    res.rhoU[axis] = -res.rhoU[axis];
                    return res;
                };
            }
            else if (type == BCType::Outflow)
            {
                m_bc_real[face_idx] = [](PetscReal, const State& st) { return st; };
                m_bc_cplx[face_idx] = [](PetscReal, const CplxState& st) { return st; };
            }
            else if (type != BCType::Periodic)
            {
                std::println("Error: Use the lambda or fixed-state overload for this BC type!");
                std::terminate();
            }
        }

        void bind_lambda_bc(Direction dir, BCType type, ValidFactoryBC<T, Dims> auto factory_lambda)
        {
            auto face_idx       = static_cast<std::size_t>(dir);
            m_types[face_idx]   = type;
            m_bc_real[face_idx] = factory_lambda;

            m_bc_cplx[face_idx] = [factory_lambda](PetscReal t, const CplxState& cplx_st)
            {
                // 1. Isolate real state
                State real_st;
                real_st.rho  = std::real(cplx_st.rho);
                real_st.rhoE = std::real(cplx_st.rhoE);
                for (std::size_t i = 0; i < Dims; ++i)
                    real_st.rhoU[i] = std::real(cplx_st.rhoU[i]);

                // 2. Eval factory lambda
                State real_res = factory_lambda(t, real_st);

                // 3. Cast to Complex
                CplxState res_cplx;
                res_cplx.rho  = CplxType{real_res.rho};
                res_cplx.rhoE = CplxType{real_res.rhoE};
                for (std::size_t i = 0; i < Dims; ++i)
                    res_cplx.rhoU[i] = CplxType{real_res.rhoU[i]};

                return res_cplx;
            };
        }

        void bind_fixed_dirichlet(Direction dir, const State& fixed_state)
        {
            auto face_idx       = static_cast<std::size_t>(dir);
            m_types[face_idx]   = BCType::Dirichlet;
            m_bc_real[face_idx] = [fixed_state](PetscReal, const auto&) { return fixed_state; };

            m_bc_cplx[face_idx] = [fixed_state](PetscReal, const auto&)
            {
                CplxState res_cplx;
                res_cplx.rho  = CplxType{fixed_state.rho};
                res_cplx.rhoE = CplxType{fixed_state.rhoE};
                for (std::size_t i = 0; i < Dims; ++i)
                    res_cplx.rhoU[i] = CplxType{fixed_state.rhoU[i]};
                return res_cplx;
            };
        }

        // Generic Evaluator
        template <real_or_complex_floating_point U>
        ComprFD::NodeState<U, Dims> evaluate_bc(Direction dir, PetscReal t,
                                                const ComprFD::NodeState<U, Dims>& interior) const
        {
            const auto face_idx = static_cast<std::size_t>(dir);
            if constexpr (is_complex_floating_point<std::remove_const_t<U>>::value)
                return m_bc_cplx[face_idx](t, interior);
            else
                return m_bc_real[face_idx](t, interior);
        }

      public:
        // Volume Conditions (Dimension independent)
        std::function<State(PetscReal)> initial_cond;
        std::function<State(PetscReal, PetscReal)>
            exact_sol; // Note: Might need adjusting for 2D pos

        BCType get_type(Direction dir) const { return m_types[static_cast<std::size_t>(dir)]; }
        void   set_bc(const Direction dir, BCType type) { this->bind_native_bc(dir, type); }
        void   set_bc(const Direction dir, BCType type, ValidFactoryBC<T, Dims> auto func)
        {
            this->bind_lambda_bc(dir, type, func);
        }
        void set_bc(const Direction dir, const State& st) { this->bind_fixed_dirichlet(dir, st); }
    };

    // ====================================================================
    // FRONT-END CLASSES: Provide the nice dimension-specific names
    // ====================================================================

    // Fallback template
    template <std::floating_point T, std::size_t n_dims>
    struct BoundaryConditions;

    // ── 1D Specialization ───────────────────────────────────────────────
    template <std::floating_point T>
    struct BoundaryConditions<T, 1> : public BoundaryConditionsBase<T, 1>
    {
        using Base  = BoundaryConditionsBase<T, 1>;
        using State = ComprFD::NodeState<T, 1>;

        // Left (Face 0)
        void set_left_bc(BCType type) { this->bind_native_bc(Direction::Left, type); }
        void set_left_bc(BCType type, ValidFactoryBC<T, 1> auto func)
        {
            this->bind_lambda_bc(Direction::Left, type, func);
        }
        void set_left_bc(const State& st) { this->bind_fixed_dirichlet(Direction::Left, st); }

        // Right (Face 1)
        void set_right_bc(BCType type) { this->bind_native_bc(Direction::Right, type); }
        void set_right_bc(BCType type, ValidFactoryBC<T, 1> auto func)
        {
            this->bind_lambda_bc(Direction::Right, type, func);
        }
        void set_right_bc(const State& st) { this->bind_fixed_dirichlet(Direction::Right, st); }

        // Getters
        template <real_or_complex_floating_point U>
        auto get_left_bc(PetscReal t, const ComprFD::NodeState<U, 1>& int_st) const
        {
            return this->evaluate_bc(Direction::Left, t, int_st);
        }

        template <real_or_complex_floating_point U>
        auto get_right_bc(PetscReal t, const ComprFD::NodeState<U, 1>& int_st) const
        {
            return this->evaluate_bc(Direction::Right, t, int_st);
        }
    };

    // ── 2D Specialization ───────────────────────────────────────────────
    template <std::floating_point T>
    struct BoundaryConditions<T, 2>
    {
        using State     = ComprFD::NodeState<T, 2>;
        using CplxState = ComprFD::NodeState<CplxType, 2>;

        using RealFunc = std::function<State(PetscReal, PetscReal, PetscReal, const State&)>;
        using CplxFunc =
            std::function<CplxState(PetscReal, PetscReal, PetscReal, const CplxState&)>;

        std::array<BCType, 4>   m_types;
        std::array<RealFunc, 4> m_bc_real;
        std::array<CplxFunc, 4> m_bc_cplx;

        std::function<State(T, T)>    initial_cond;
        std::function<State(T, T, T)> exact_sol;

        BoundaryConditions()
        {
            for (std::size_t i = 0; i < 4; ++i)
            {
                bind_native(static_cast<Direction>(i), BCType::Outflow);
            }
        }

        template <real_or_complex_floating_point U>
        ComprFD::NodeState<U, 2> evaluate_bc(Direction dir, PetscReal t, PetscReal x, PetscReal y,
                                             const ComprFD::NodeState<U, 2>& interior) const
        {
            const auto face_idx = static_cast<std::size_t>(dir);
            if constexpr (is_complex_floating_point<std::remove_const_t<U>>::value)
                return m_bc_cplx[face_idx](t, x, y, interior);
            else
                return m_bc_real[face_idx](t, x, y, interior);
        }

        // ── West / Left (Face 0) ──
        void set_left_bc(BCType type) { bind_native(Direction::Left, type); }
        void set_left_bc(RealFunc func) { bind_lambda(Direction::Left, func); } // FIXED

        // ── East / Right (Face 1) ──
        void set_right_bc(BCType type) { bind_native(Direction::Right, type); }
        void set_right_bc(RealFunc func) { bind_lambda(Direction::Right, func); } // FIXED

        // ── South / Bottom (Face 2) ──
        void set_bottom_bc(BCType type) { bind_native(Direction::Bottom, type); }
        void set_bottom_bc(RealFunc func) { bind_lambda(Direction::Bottom, func); } // FIXED

        // ── North / Top (Face 3) ──
        void set_top_bc(BCType type) { bind_native(Direction::Top, type); }
        void set_top_bc(RealFunc func) { bind_lambda(Direction::Top, func); } // FIXED

        BCType get_type(Direction dir) const { return m_types[static_cast<std::size_t>(dir)]; }

      private:
        // NEW: Handles complex unwrapping for custom lambda expressions
        void bind_lambda(Direction dir, RealFunc func)
        {
            auto idx       = static_cast<std::size_t>(dir);
            m_types[idx]   = BCType::Custom;
            m_bc_real[idx] = func;

            m_bc_cplx[idx] = [func](PetscReal t, PetscReal x, PetscReal y, const CplxState& cplx_st)
            {
                State real_st;
                real_st.rho  = std::real(cplx_st.rho);
                real_st.rhoE = std::real(cplx_st.rhoE);
                for (std::size_t i = 0; i < 2; ++i)
                    real_st.rhoU[i] = std::real(cplx_st.rhoU[i]);

                State real_res = func(t, x, y, real_st);

                CplxState res_cplx;
                res_cplx.rho  = CplxType{real_res.rho};
                res_cplx.rhoE = CplxType{real_res.rhoE};
                for (std::size_t i = 0; i < 2; ++i)
                    res_cplx.rhoU[i] = CplxType{real_res.rhoU[i]};

                return res_cplx;
            };
        }

        void bind_native(Direction dir, BCType type)
        {
            auto idx     = static_cast<std::size_t>(dir);
            m_types[idx] = type;
            if (type == BCType::Wall)
            {
                const std::size_t axis = idx / 2;
                m_bc_real[idx]         = [axis](PetscReal, PetscReal, PetscReal, const State& st)
                {
                    State res      = st;
                    res.rhoU[axis] = -res.rhoU[axis];
                    return res;
                };
                m_bc_cplx[idx] = [axis](PetscReal, PetscReal, PetscReal, const CplxState& st)
                {
                    CplxState res  = st;
                    res.rhoU[axis] = -res.rhoU[axis];
                    return res;
                };
            }
            else if (type == BCType::Outflow || type == BCType::Periodic)
            {
                m_bc_real[idx] = [](PetscReal, PetscReal, PetscReal, const State& st)
                { return st; };
                m_bc_cplx[idx] = [](PetscReal, PetscReal, PetscReal, const CplxState& st)
                { return st; };
            }
        }
    };

} // namespace numPDE
