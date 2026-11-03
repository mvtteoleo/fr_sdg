#include <iostream>
#include <print>
#include <ranges>
#include <vector>

using Vec = std::vector<double>;

template <std::ranges::random_access_range R>
auto square_line(R r_in)
{
    return r_in | std::views::transform([](auto i) { return i * i; });
}

struct Mat
{
    Vec&        data;
    std::size_t rows;
    std::size_t cols;
};

auto isolate_col(Mat& mat, std::integral auto target_col)
{
    return std::views::iota(0uz, mat.cols) |
           std::views::transform([&mat, target_col](std::size_t j) -> double&
                                 { return mat.data[j * mat.cols + target_col]; });
}

void print_mat(const Mat& mat, std::string_view name)
{
    std::println("{}:", name);
    for (std::size_t i = 0; i < mat.rows; ++i)
    {
        for (std::size_t j = 0; j < mat.cols; ++j)
        {
            // {:4} forces a width of 4 characters for clean alignment
            std::print("{:4} ", mat.data[i * mat.cols + j]);
        }
        std::println(); // Newline at the end of each row
    }
    std::println();
}

int main(int argc, char* argv[])
{

    auto cols = 5uz;
    auto rows = 5uz;

    auto F_r = Vec(cols * rows, 0.0);
    auto F_l = Vec(cols * rows, 0.0);
    auto D   = Vec(cols * rows, 0.0);

    for (const auto i : std::views::iota(0uz, rows))
        for (const auto j : std::views::iota(0uz, cols))
        {
            auto idx = i * cols + j;
            F_r[idx] = i * 10 + j;
            F_l[idx] = i * 10 + j;
        }

    auto mat    = Mat{.data = F_r, .rows = cols, .cols = rows};
    auto mat_or = Mat{.data = F_l, .rows = cols, .cols = rows};

    // Print them nicely
    print_mat(mat, "M Matrix");

    auto col_0     = isolate_col(mat, 0uz);
    auto col_0_sqr = square_line(col_0);

    std::println("Original col_0");
    for (auto val : col_0)
        std::cout << val << " ";

    std::println();
    std::println("Modified col");
    for (auto val : col_0_sqr)
        std::cout << val << " ";
    std::println();

    auto col_n = isolate_col(mat, 1);
    std::ranges::copy(col_0_sqr, col_n.begin());

    print_mat(mat, "M Matrix after modifing col n");

    std::ranges::copy(col_n, col_n.begin());
    print_mat(mat, "M Matrix after restoring col n");

    auto row_0     = std::span(&mat.data[3 * mat.cols], mat.cols);
    auto row_0_sqr = square_line(row_0);
    std::ranges::copy(row_0_sqr, row_0.begin());

    print_mat(mat, "M Matrix after modifing row 3");

    std::ranges::copy(row_0, col_0.begin());
    print_mat(mat, "M Matrix after setting col_0 to row_3_sqr");
    return 0;
}
