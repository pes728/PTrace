#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"

#include <time.h>
#include <iostream>
#include <float.h>
#include <stdio.h>

#include <SDL.h>
#undef main


#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

#define num_objects 5


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);


void updateTexture(SDL_Texture* texture);

__global__ void render_init(int width, int height, curandState* rand_state);

__global__ void render(vec3* dev_pixels, int width, int height, int aa, camera** cam, hittable** world, curandState* rand_state);

__global__ void createWorld(hittable** d_list, hittable** d_world, camera** d_camera, int width, int height);

__device__ vec3 color(const ray& r, hittable** world, curandState* local_rand_state);

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera);

const int WIDTH = 2000, HEIGHT = 1000;

unsigned int aa = 100;

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
    
    std::cerr << "Rendering a " << WIDTH<< "x" << HEIGHT << " image with " << aa << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = WIDTH * HEIGHT;

    vec3* pixels;

    checkCudaErrors(cudaMallocManaged((void**)&pixels, num_pixels * sizeof(vec3)));
    
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));


    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_objects));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    createWorld <<<1, 1 >>> (d_list, d_world, d_camera, WIDTH, HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    clock_t start, stop;
    start = clock();


    dim3 blocks(WIDTH / tx + 1, HEIGHT / ty + 1);
    dim3 threads(tx, ty);

    render_init <<<blocks, threads>>> (WIDTH, HEIGHT, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render <<<blocks, threads >>>(pixels, WIDTH, HEIGHT, aa, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    stop = clock();
    
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

    std::cerr << "took " << timer_seconds << " seconds.\n";


    Uint32* fb = new Uint32[WIDTH * HEIGHT];

    //magic vec3 to uint32
    for (int i = 0; i < WIDTH * HEIGHT; i++) fb[i] = 0x000000FF | uint8_t(255.99 * pixels[i].e[0]) | uint8_t(255.99 * pixels[i].e[1]) << 8 | uint8_t(255.99 * pixels[i].e[2]) << 16;

    SDL_UpdateTexture(texture, NULL, fb, WIDTH * sizeof(Uint32));

    checkCudaErrors(cudaDeviceSynchronize());
    free_world <<<1, 1>>> (d_list, d_world, d_camera);
    
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(pixels));

    cudaDeviceReset();
}

__global__ void render_init(int width, int height, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height)) return;

    int pixel_index = j * width + i;

    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void createWorld(hittable** d_list, hittable** d_world, camera** d_camera, int width, int height) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
            new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
            new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5,
            new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
            new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1, 0, -1), -0.45,
            new dielectric(1.5));

        *d_world = new hittable_list(d_list, num_objects);
        *d_camera = new camera(vec3(-2, 2, 1), vec3(0, 0, -1), vec3(0, 1, 0), 20.0, float(width) / float(height));
    }
}

__global__ void render(vec3* pixels, int width, int height, int aa, camera** cam, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height)) return;

    int pixelIndex = j * width + i;
    
    curandState local_rand_state = rand_state[pixelIndex];

    vec3 col;

    for (int s = 0; s < aa; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(width);
        float v = float(j + curand_uniform(&local_rand_state)) / float(height);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixelIndex] = local_rand_state;
    col /= float(aa);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    pixels[pixelIndex] = col;
}

__device__ vec3 color(const ray& r, hittable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3();
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.d);
            float t = 0.5f * (unit_direction[1] + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3();
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int width, int height) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
            new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
            new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5,
            new metal(vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
            new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1, 0, -1), -0.45,
            new dielectric(1.5));

        *d_world = new hittable_list(d_list, num_objects);
        *d_camera = new camera(vec3(-2, 2, 1), vec3(0, 0, -1), vec3(0, 1, 0), 20.0, float(width) / float(height));
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i = 0; i < num_objects; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete* d_camera;
}

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error: " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}
