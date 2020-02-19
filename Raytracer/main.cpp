#include "Raytracer.hpp"

void populateGeometry(std::vector<std::shared_ptr<Shape>>& shapes)
{
	for (size_t i{ 3 }; i < 7; i++)
	{
		Sphere s{ { -350 + (60.0f * i), 0, 0 }, 10.0f + (10.0f * i) };
		s.setColour({ 0.1f, 0.1f * i, 0.1f });

		shapes.push_back(std::make_shared<Sphere>(s));
	}
	
	Plane p0{ { 0, 0, -300 }, { 0, 0.5f, 1 } };
	p0.setColour({ 0.1f, 0.5f, 0.5f });

	shapes.push_back(std::make_shared<Plane>(p0));

	Plane p1{ { 0, 0, -300 }, { 0, -0.5f, 1 } };
	p1.setColour({ 0.5f, 0.1f, 0.5f });
	
	shapes.push_back(std::make_shared<Plane>(p1));
	
	Triangle t0{ { 0, 100, -10 },
				{ 20, 100, -10 },
				{ 10, 80, -10 } };
	t0.setColour({ 0.1f, 0.1f, 0.5f });

	shapes.push_back(std::make_shared<Triangle>(t0));

	Triangle t1{ { 0, -100, -10 },
				{ 20, -100, -10 },
				{ 10, -80, -10 } };
	t1.setColour({ 0.1f, 0.1f, 0.5f });

	shapes.push_back(std::make_shared<Triangle>(t1));

}

int main()
{
	AmbientLight ambLight{ Colour{ 0.2f, 0.2f, 0.2f } };

	Camera camera;
	camera.setEye({ -175,0,130 });
	camera.setLookAt({ -100,0,0 });
	camera.setFrustrumDistance(280.0f);
	camera.computeUVW();

	const std::size_t imageWidth{ 600 };
	const std::size_t imageHeight{ 600 };

	Ray ray;
	ray.o = camera.getmEye();
	std::vector<std::shared_ptr<Shape>> shapes;
	populateGeometry(shapes);

	std::vector<Colour> image(imageWidth * imageHeight);

	for (std::size_t y{ 0 }; y < imageHeight; ++y)
	{
		for (std::size_t x{ 0 }; x < imageWidth; ++x)
		{
			Colour pixel{ 0, 0, 0 };
			/*
			for (std::size_t py{ 0 }; py < 4; py++)
			{
				for (std::size_t px{ 0 }; px < 4; px++)
				{
				*/
					float depth = FLT_MAX;
					// Compute origin of ray. 
					// -0.5 to get the corner of each pixel then go through each grid corner and add random range of grid section size (1 / 4)
					//float originX = (x - 0.5f * (imageWidth - 1.0f)) - 0.5f + (px * 0.25f) + randomRange(0, 0.25f);
					//float originY = (y - 0.5f * (imageHeight - 1.0f)) - 0.5f + (py * 0.25f) + randomRange(0, 0.25f);
					float originX = (x - 0.5f * (imageWidth - 1.0f));
					float originY = (y - 0.5f * (imageHeight - 1.0f));

					ray.d = glm::normalize((originX * camera.getmV()) + (originY * camera.getmU() - (camera.getFrustrumDistance() * camera.getmW())));
					//std::cout << ray.d.x << " : " << ray.d.y << " : " << ray.d.z << std::endl;
					for (std::size_t i{ 0 }; i < shapes.size(); i++)
					{
						std::pair<float, std::shared_ptr<Material>> testData = shapes[i]->hit(ray);
						if (testData.first < depth)
						{
							depth = testData.first;
							pixel = ambLight.getColour() + testData.second->shade(ray.eval(depth));
						}
					}
					/*
				}
			}

			// Average
			pixel.r /= 16.0f;
			pixel.g /= 16.0f;
			pixel.b /= 16.0f;
			*/
			image[x + y * imageHeight] = pixel;
		}
		std::cout << y << std::endl;
	}

	saveToFile("sphere.bmp", imageWidth, imageHeight, image);
	return 0;
}

void Light::setColour(Colour colour)
{
	this->colour = colour;
}

void Light::setRadiance(float radiance)
{
	this->radiance = radiance;
}

Colour AmbientLight::getColour()
{
	return colour;
}

// ***** Camera function members *****
Camera::Camera() :
	mEye{ 0.0f, 0.0f, 100.0f },
	mLookAt{ 0.0f },
	mUp{ 0.0f, 1.0f, 0.0f },
	mU{ 1.0f, 0.0f, 0.0f },
	mV{ 0.0f, 1.0f, 0.0f },
	mW{ 0.0f, 0.0f, 1.0f }
{}

void Camera::setEye(Point const& eye)
{
	mEye = eye;
}

void Camera::setLookAt(Point const& lookAt)
{
	mLookAt = lookAt;
}

void Camera::setUpVector(Vector const& up)
{
	mUp = up;
}

void Camera::setFrustrumDistance(float frustumDistance)
{
	mFrustrumDistance = frustumDistance;
}

const float& Camera::getFrustrumDistance()
{
	return mFrustrumDistance;
}

void Camera::computeUVW()
{
	mW = (mEye - mLookAt) / glm::length(mEye - mLookAt);
	mV = glm::cross(mW, mUp) / glm::length(glm::cross(mW, mUp));
	mU = glm::cross(mW, mV);
}

const Point& Camera::getmEye()
{
	return mEye;
}

const Vector& Camera::getmW()
{
	return mW;
}

const Vector& Camera::getmV()
{
	return mV;
}

const Vector& Camera::getmU()
{
	return mU;
}


// ***** Shape function members *****
Shape::Shape() : mColour{ 0, 0, 0 }
{}

void Shape::setColour(Colour const& col)
{
	mColour = col;
}

Colour Shape::getColour() const
{
	return mColour;
}

// ***** Sphere function members *****
Sphere::Sphere(Point center, float radius) :
	mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

std::pair<float, std::shared_ptr<Material>> Sphere::hit(Ray const& ray) const
{
	Vector temp = ray.o - mCentre;
	float a = glm::dot(ray.d, ray.d);
	float b = 2.0f * glm::dot(temp, ray.d);
	float c = glm::dot(temp, temp) - mRadiusSqr;
	float disc = b * b - 4.0f * a * c;

	float t = -b - glm::sqrt(disc);

	if (disc < 0.0f)
	{
		return std::make_pair(FLT_MAX, mMaterial);
	}

	return std::make_pair(t, mMaterial);
}

// ***** Plane function members *****
Plane::Plane(Point point, Vector normal) :
	point{ point }, normal{ normal }
{}

Colour Matte::shade(Point surfacePoint)
{
	return Colour();
}

std::pair<float, std::shared_ptr<Material>> Plane::hit(Ray const& ray) const
{
	float denom = glm::dot(normal, ray.d);
	if (abs(denom) < 0.000001f) return std::make_pair(FLT_MAX, mMaterial);

	float t = glm::dot((point - ray.o), normal) / denom;
	if (t > 0)
	{
		return std::make_pair(t, mMaterial);
	}

	return std::make_pair(FLT_MAX, mMaterial);
}


// ***** Triangle function members *****
Triangle::Triangle(Point point1, Point point2, Point point3) :
	p0{ point1 }, p1{ point2 }, p2{ point3 }
{}

std::pair<float, std::shared_ptr<Material>> Triangle::hit(Ray const& ray) const
{
	const double ep = 0.000001;
	glm::vec3 v2v0 = p2 - p0;
	glm::vec3 v1v0 = p1 - p0;
	glm::vec3 rayv0 = ray.o - p0;
	glm::vec3 pvec = glm::cross(ray.d, v2v0);

	float det = glm::dot(v1v0, pvec);
	float invDet = 1.0f / det;

	float u = glm::dot(rayv0, pvec) * invDet;

	if (u < 0 || u > 1)
	{
		return std::make_pair(FLT_MAX, mMaterial);
	}

	glm::vec3 qvec = glm::cross(rayv0, v1v0);

	float v = glm::dot(ray.d, qvec) * invDet;

	if (v < 0 || u + v > 1)
	{
		return std::make_pair(FLT_MAX, mMaterial);
	}

	float t = glm::dot(v2v0, qvec) * invDet;

	return std::make_pair(t, mMaterial);
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