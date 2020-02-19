#pragma once

#include <glm/glm.hpp>
#include <limits>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

#include <vector>

using Point = glm::vec3;
using Vector = glm::vec3;
using Colour = glm::vec3;

// Declarations
class Camera;
class Shape;
class Sampler;
class Light;

class Light
{
public:
	Light(Colour colour = { 0,0,0 }) : colour(colour), radiance(1) {};

	virtual ~Light() = default;

	void setColour(Colour colour);
	void setRadiance(float radiance);
protected:
	Colour colour;
	float radiance = 1;
};

class PointLight : public Light
{
public:
	PointLight();

protected:
	Point point;
};

class AmbientLight : public Light
{
public:
	AmbientLight(Colour colour);

	Colour getColour();
};

class Camera
{
public:
	Camera();

	virtual ~Camera() = default;

	void setEye(Point const& eye);

	void setLookAt(Point const& lookAt);

	void setUpVector(Vector const& up);

	void setFrustrumDistance(float frustumDistance);

	void computeUVW();

	const Point& getmEye();

	const Vector& getmW();
	const Vector& getmV();
	const Vector& getmU();
	const float& getFrustrumDistance();
protected:
	Point mEye;
	Point mLookAt;
	Point mUp;
	Vector mU, mV, mW;
	float mFrustrumDistance = 100;
};

struct Ray
{
	Point o;
	Vector d;

	Point eval(float t)
	{
		return o + t * d;
	}
};


class Shape
{
public:
	Shape();
	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be updated in sr
	virtual std::pair<float, std::shared_ptr<Material>> hit(Ray const& ray) const = 0;

	void setColour(Colour const& col);

	Colour getColour() const;

protected:
	Colour mColour;
	std::shared_ptr<Material> mMaterial;
};

class Material
{
public:
	virtual ~Material() = default;

	virtual Colour shade(Point surfacePoint) = 0;
};


class Matte : public Material
{
public:
	Colour shade(Point surfacePoint);
};

// Concrete classes which we can construct and use in our ray tracer

class Sphere : public Shape
{
public:
	Sphere(Point center, float radius);

	std::pair<float, std::shared_ptr<Material>> hit(Ray const& ray) const;

private:
	Point mCentre;
	float mRadius;
	float mRadiusSqr;
};

class Plane : public Shape
{
public:
	Plane(Point point, Vector normal);

	std::pair<float, std::shared_ptr<Material>> hit(Ray const& ray) const;

private:
	Point point;
	Vector normal;
};

class Triangle : public Shape
{
public:
	Triangle(Point p0, Point p1, Point p2);

	std::pair<float, std::shared_ptr<Material>> hit(Ray const& ray) const;

private:
	Point p0, p1, p2;
};

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

void populateGeometry(std::vector<Shape>& shapes);

float randomRange(float min, float max);