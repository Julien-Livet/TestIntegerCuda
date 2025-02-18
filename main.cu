#include <iostream>

#include "Integer.cuh"

int main()
{
    using Z = Integer<unsigned long long, std::vector<unsigned long long> >;
    
    {
        std::string const s("56062005704198360319209");
        Z const n(s);
        
        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }

    {
        std::string const s("4113101149215104800030529537915953170486139623539759933135949994882770404074832568499");
        Z const n(s);

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }

    {
        Z n;
        n.setPrecision(1024 / 64);
        std::cout << "hey0" << std::endl;
        n.setRandom<std::random_device>();
        std::cout << "hey1" << std::endl;
        n.setPositive();
        if (!(n % 2))
            ++n;

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }

    return 0;
}
