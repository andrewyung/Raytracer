#include "Raytracer.hpp"

void populateGeometry(std::vector<Sphere>& spheres, std::vector<Plane>& planes)
{
	Sphere s;
	s.centre = { -350, 0, 0 };
	s.radius = 10.0f;
	for (size_t i{ 0 }; i < 7; i++)
	{
		s.centre = { s.centre.x + 80, 0, 0 };
		s.radius += 10.0f;

		spheres.push_back(s);
	}

	Plane p;
	p.point = { 0, 0, 0 };
	p.normal = { 0, 1, 1 };

	planes.push_back(p);

	p.point = { 0, 0, 0 };
	p.normal = { 0, -0.5, 1 };

	planes.push_back(p);
}

int main()
{
	const std::size_t imageWidth{ 600 };
	const std::size_t imageHeight{ 600 };

	Ray ray;
	ray.d = { 0, 0, -1 };
	std::vector<Sphere> spheres;
	std::vector<Plane> planes;
	populateGeometry(spheres, planes);

	Triangle triangle;
	triangle.v0 = { 0,0,0 };
	triangle.v1 = { 100,0,0 };
	triangle.v2 = { 100,100,0 };

	std::vector<Colour> image(imageWidth * imageHeight);

	for (std::size_t y{ 0 }; y < imageHeight; ++y)
	{
		for (std::size_t x{ 0 }; x < imageWidth; ++x)
		{
			Colour pixel{ 0, 0, 0 };
			for (std::size_t py{ 0 }; py < 4; py++)
			{
				for (std::size_t px{ 0 }; px < 4; px++)
				{
					// Compute origin of ray. 
					// -0.5 to get the corner of each pixel then go through each grid corner and add random range of grid section size (1 / 4)
					float originX = (x - 0.5f * (imageWidth - 1.0f)) - 0.5f + (px * 0.25f) + randomRange(0, 0.25f);
					float originY = (y - 0.5f * (imageHeight - 1.0f)) - 0.5f + (py * 0.25f) + randomRange(0, 0.25f);

					ray.o = { originX, originY, 100.0f };

					for (std::size_t i{ 0 }; i < spheres.size(); i++)
					{
						pixel += intersectRayWithSphere(spheres[i], ray, Colour(0.5f, 0, 0));
					}
					for (std::size_t i{ 0 }; i < planes.size(); i++)
					{
						pixel += intersectRayWithPlane(planes[i], ray, Colour(0, 0.5f, 0));
					}
				}
			}

			// Average
			pixel.r /= 16.0f;
			pixel.g /= 16.0f;
			pixel.b /= 16.0f;

			image[x + y * imageHeight] = pixel;
		}
		std::cout << y << std::endl;
	}

	saveToFile("sphere.bmp", imageWidth, imageHeight, image);
	return 0;
}

Colour intersectRayWithSphere(Sphere s, Ray r, Colour colour)
{
	Vector temp = r.o - s.centre;
	float a = glm::dot(r.d, r.d);
	float b = 2.0f * glm::dot(temp, r.d);
	float c = glm::dot(temp, temp) - (s.radius * s.radius);
	float disc = b * b - 4.0f * a * c;

	if (disc < 0.0f)
	{
		return { 0, 0, 0 };
	}

	return colour;
}

Colour intersectRayWithPlane(Plane p, Ray r, Colour colour)
{
	float denom = glm::dot(p.normal, r.d);
	if (abs(denom) < 0.000001f) return { 0, 0, 0 };

	float t = glm::dot((p.point - r.o), p.normal) / denom;
	if (t > 0)
	{
		return colour;
	}

	return { 0, 0, 0 };
}


Colour intersectRayWithTriangle(Triangle t, Ray r, Colour colour)
{
	Vector v0v1 = t.v0 - t.v1;
	Vector v0v2 = t.v0 - t.v2;
	Vector v0ray = t.v0 - r.o;

	glm::mat3 dInv = glm::inverse(glm::mat3(v0v1, v0v2, r.d));

	double beta = glm::determinant(glm::mat3(v0ray, v0v2, r.d) * dInv);

	if (beta < 0.0 || beta > 1) return { 0, 0, 0 };

	double gamma = glm::determinant(glm::mat3(v0v1, v0ray, r.d) * dInv);

	if (gamma < 0.0 || gamma > 1) return { 0, 0, 0 };

	if (beta + gamma > 1.0) return { 0, 0, 0 };

	return colour;
}


void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image)
{
	std::vector<unsigned char> data(image.size() * 3);

	for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
	{
		Colour pixel = image[i];
		data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
		data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
		data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
	}

	stbi_write_bmp(filename.c_str(),
		static_cast<int>(width),
		static_cast<int>(height),
		3,
		data.data());
}

float randomRange(float min, float max)
{
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}