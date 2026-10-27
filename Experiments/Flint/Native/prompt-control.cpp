#include "FlashFlint.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    std::ifstream stream(argv[1]); const std::string input((std::istreambuf_iterator<char>(stream)), {});
    const int n = flf_render_prompt(input.data(), input.size(), nullptr, 0);
    if (n >= 0) { std::cerr << flf_last_error() << std::endl; return 1; }
    std::vector<char> output(-n);
    const int size = flf_render_prompt(input.data(), input.size(), output.data(), output.size());
    if (size != int(output.size())) { std::cerr << flf_last_error() << std::endl; return 1; }
    std::cout.write(output.data(), output.size()); return 0;
}
