#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <time.h>
#include <iostream>
#include "vec3.h"
#include <SDL.h>
#undef main

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);


void updateTexture(SDL_Texture* texture);

__global__ void render(vec3* dev_pixels, int width, int height);


const int WIDTH = 2000, HEIGHT = 1000;

unsigned int aa = 16;

int max_depth = 4;

int tx = 8;
int ty = 8;


int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
    }

    SDL_Window* window = SDL_CreateWindow("PTrace", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_ALLOW_HIGHDPI);
    if (window == NULL) {
        std::cerr << "Could not create window:" << SDL_GetError() << std::endl;
        return EXIT_FAILURE;
    }

    SDL_Renderer* renderer;
    SDL_Texture* texture;

    renderer = SDL_CreateRenderer(window, -1, 0);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);

    /*scene.add(std::make_shared<sphere>(vec3(0, 0, -1.0), 0.5, std::make_shared<lambertian>(vec3(0.7, 0.3, 0.0))));

    scene.add(std::make_shared<sphere>(vec3(1.0, 0, -1.0), 0.5, std::make_shared<metal>(vec3(0.8, 0.6, 0.2), 0.3)));

    scene.add(std::make_shared<sphere>(vec3(-1.0, 0, -1.0), 0.5, std::make_shared<metal>(vec3(0.8, 0.8, 0.0), 1.0)));

    scene.add(std::make_shared<sphere>(vec3(0, -100.5, 0), 100, std::make_shared<lambertian>(vec3(0.8, 0.8, 0.0))));*/


    updateTexture(texture);


    SDL_Event windowEvent;
    while (1) {

        SDL_Delay(500);

        if (SDL_PollEvent(&windowEvent)) {
            if (windowEvent.type == SDL_QUIT) break;

            if (windowEvent.type == SDL_KEYDOWN) {
                switch (windowEvent.key.keysym.sym) {
                case SDLK_RETURN:
                    updateTexture(texture);
                    break;
                }
            }
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopyEx(renderer, texture, NULL, NULL, 0.0, NULL, SDL_FLIP_VERTICAL);
        SDL_RenderPresent(renderer);

    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return EXIT_SUCCESS;
}

void updateTexture(SDL_Texture* texture) {
    
    vec3* pixels;

    pixels = new vec3[WIDTH * HEIGHT];

    checkCudaErrors(cudaMallocManaged((void**)&pixels, WIDTH * HEIGHT * sizeof(vec3)));
    

    clock_t start, stop;
    start = clock();


    dim3 blocks(WIDTH / tx + 1, HEIGHT / ty + 1);
    dim3 threads(tx, ty);

    render <<<blocks, threads >>>(pixels, WIDTH, HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    stop = clock();
    
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

    std::cerr << "took " << timer_seconds << " seconds.\n";


    Uint32* fb = new Uint32[WIDTH * HEIGHT];

    //magic vec3 to uint32
    for (int i = 0; i < WIDTH * HEIGHT; i++) fb[i] = 0x000000FF | uint8_t(255.99 * pixels[i].e[0]) | uint8_t(255.99 * pixels[i].e[1]) << 8 | uint8_t(255.99 * pixels[i].e[2]) << 16;

    SDL_UpdateTexture(texture, NULL, fb, WIDTH * sizeof(Uint32));
}

__global__ void render(vec3* d_pixels, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;
    int pixelIndex = j * width + i;
    d_pixels[pixelIndex] = vec3(float(i) / width, float(j) / height, 0.2f);
}

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}
