#include "hittable_list.h"
#include "sphere.h"
#include "color.h"
#include <iostream>
#include "camera.h"
#include "PMath.h"
#include "material.h"
#include <SDL.h>
#undef main


const int WIDTH = 2000, HEIGHT = 1000;

unsigned int aa = 4;

int max_depth = 4;

Uint32* pixels;
SDL_Renderer* renderer;
SDL_Texture* texture;

void updateTexture(const hittable_list& scene, camera& cam);

vec3 ray_color(const ray& r, const hittable_list& scene, int depth);

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

    renderer = SDL_CreateRenderer(window, -1, 0);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);

    pixels = new Uint32[WIDTH * HEIGHT];


    camera cam;

    hittable_list scene;

    scene.add(std::make_shared<sphere>(vec3(0, 0, -1.0), 0.5, std::make_shared<lambertian>(vec3(0.7, 0.3, 0.0))));

    scene.add(std::make_shared<sphere>(vec3(1.0, 0, -1.0), 0.5, std::make_shared<metal>(vec3(0.8, 0.6, 0.2), 0.3)));
    
    scene.add(std::make_shared<sphere>(vec3(-1.0, 0, -1.0), 0.5, std::make_shared<metal>(vec3(0.8, 0.8, 0.0), 1.0)));

    scene.add(std::make_shared<sphere>(vec3(0, -100.5, 0), 100, std::make_shared<lambertian>(vec3(0.8, 0.8, 0.0))));


    updateTexture(scene, cam);


    SDL_Event windowEvent;
    while (1) {

        SDL_Delay(500);

        if (SDL_PollEvent(&windowEvent)) {
            if (windowEvent.type == SDL_QUIT) break;

            if (windowEvent.type == SDL_KEYDOWN) {
                switch (windowEvent.key.keysym.sym) {
                case SDLK_RETURN:
                    updateTexture(scene, cam);
                    break;
                }
            }
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopyEx(renderer, texture, NULL, NULL, 0.0, NULL, SDL_FLIP_VERTICAL);
        SDL_RenderPresent(renderer);

    }

    delete[] pixels;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return EXIT_SUCCESS;
}

void updateTexture(const hittable_list& scene, camera& cam) {
    for (int y = 0; y < HEIGHT; y++) {
        std::cout << "scanlines left: " << HEIGHT - y << std::endl;
        for (int x = 0; x < WIDTH; x++) {

            vec3 color;
            for (int s = 0; s < aa; s++) {
                auto u = double(x + random_double()) / WIDTH;
                auto v = double(y + random_double()) / HEIGHT;
                ray r = cam.get_ray(u, v);

                hit_record rec;

                color += ray_color(r, scene, max_depth);
            }
            pixels[y * WIDTH + x] = color.to_color(aa).getColor();
        }
    }
    std::cout << "All done!" << std::endl;
    SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(Uint32));
}

vec3 ray_color(const ray& r, const hittable_list& scene, int depth)
{
    if(depth <= 0)return vec3();
 
    hit_record rec;
    
    if (scene.hit(r, 0.001, INFINITY, rec)) {
        ray scattered;
        vec3 attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            if (scattered.d == vec3()) {
                return attenuation;
            }
            return attenuation * ray_color(scattered, scene, depth - 1);
        }
        return vec3();

    }
    return ray_to_background(r);
}
