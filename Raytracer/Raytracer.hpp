#pragma once

#include <glm/glm.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

#include <vector>

using Point = glm::vec3;
using Vector = glm::vec3;
using Colour = glm::vec3;

struct Ray
{
	Point o;
	Vector d;

	Point eval(float t)
	{
		return o + t * d;
	}
};

struct Sphere
{
	Point centre;
	float radius;
};

struct Plane
{
	Point point;
	Vector normal;
};

struct Triangle
{
	Point v0;
	Point v1;
	Point v2;
};

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

void populateGeometry(std::vector<Sphere>& spheres,
	std::vector<Plane>& planes);

float randomRange(float min, float max);

Colour intersectRayWithSphere(Sphere s, Ray r, Colour colour = { 1, 0, 0 });
Colour intersectRayWithPlane(Plane p, Ray r, Colour colour = { 0, 1, 0 });
Colour intersectRayWithTriangle(Triangle t, Ray r, Colour colour = { 0, 0, 1 });