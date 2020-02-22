#include "Raytracer.hpp"

void populateGeometry(std::vector<std::shared_ptr<Shape>>& shapes)
{
	std::shared_ptr<Matte> matteMatRed = std::make_shared<Matte>(Matte{ Colour{1.0f, 0.4f, 0.2f} });

	std::shared_ptr<Matte> matteMatGreen = std::make_shared<Matte>(Matte{ Colour{0.2f, 1.0f, 0.2f} });

	std::shared_ptr<Matte> matteMatBlue = std::make_shared<Matte>(Matte{ Colour{0.2f, 0.4f, 1.0f} });

	for (size_t i{ 3 }; i < 6; i++)
	{
		Sphere s{ { -350 + (60.0f * i), 0, 0 }, 10.0f + (10.0f * i) };
		s.setColour({ 0.1f, 0.1f * i, 0.1f });
		s.setMaterial(matteMatRed);
		shapes.push_back(std::make_shared<Sphere>(s));
	}
	
	Plane p0{ { 0, 0, -300 }, { 0, 0, 1 } };
	p0.setMaterial(matteMatGreen);
	p0.setColour({ 0.1f, 0.5f, 0.5f });

	shapes.push_back(std::make_shared<Plane>(p0));

	Plane p1{ { 0, -50, 0 }, { 0.1f, 1, 0.05f } };
	p1.setMaterial(matteMatBlue);
	p1.setColour({ 0.5f, 0.1f, 0.5f });
	
	shapes.push_back(std::make_shared<Plane>(p1));
	
	/*
	Triangle t0{ { 0, 100, -10 },
				{ 20, 100, -10 },
				{ 10, 80, -10 } };
	t0.setColour({ 0.1f, 0.1f, 0.5f });
	t0.setMaterial(matteMatRed);

	shapes.push_back(std::make_shared<Triangle>(t0));

	Triangle t1{ { 0, -100, -10 },
				{ 20, -100, -10 },
				{ 10, -80, -10 } };
	t1.setColour({ 0.1f, 0.1f, 0.5f });
	t1.setMaterial(matteMatRed);

	shapes.push_back(std::make_shared<Triangle>(t1));
	*/
}

int main()
{
	std::shared_ptr<DirectionalLight> dirLight = std::make_shared<DirectionalLight>(DirectionalLight(Colour{ 1.0f, 1.0f, 1.0f }, glm::normalize(Vector{ 0.0f, 2.0f, 1.0f })));
	dirLight->setRadiance(1.5f);

	std::shared_ptr<PointLight> pointLight = std::make_shared<PointLight>(PointLight(Colour{ 1.0f, 1.0f, 1.0f }, Point{ -450.0f, 200.0f, 100.0f }));
	pointLight->setRadiance(0.003f);

	std::vector<std::shared_ptr<Light>> lights;
	//lights.push_back(dirLight);
	lights.push_back(pointLight);

	AmbientLight ambLight{ Colour{ 0.1f, 0.1f, 0.1f } };

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
					SurfaceData sd;
					sd.ambient = std::make_shared<Light>(ambLight);
					sd.lights = std::make_shared<std::vector<std::shared_ptr<Light>>>(lights);
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
						if (shapes[i]->hit(ray, sd))
						{
							pixel = sd.material->shade(ray, sd);
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
Light::Light(Colour colour) : colour(colour), radiance(1) {}
Vector Light::getDirection(SurfaceData sd)
{
	return { 0,0,0 };
}
float Light::getRadiance(SurfaceData sd)
{
	return radiance;
}
PointLight::PointLight(Colour colour, Point point) : Light(colour), point(point) {}
float PointLight::getRadiance(SurfaceData sd)
{
	return radiance * glm::clamp(glm::distance(sd.intersection, point), 0.0f, 1.0f);
}
Vector PointLight::getDirection(SurfaceData sd)
{
	return point - sd.intersection;
}

AmbientLight::AmbientLight(Colour colour) : Light(colour) {}

Vector AmbientLight::getDirection(SurfaceData sd)
{
	return { 0,0,0 };
}
DirectionalLight::DirectionalLight(Colour colour, Vector direction) : Light(colour), direction(direction) {}
Vector DirectionalLight::getDirection(SurfaceData sd)
{	
	return direction;
}
void Light::setRadiance(float radiance) 
{
	this->radiance = radiance;
}

Colour Light::getColour()
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

Material::Material(Colour diffuseColour) : diffuseColour(diffuseColour) {}
// ***** Shape function members *****
Shape::Shape() : mColour({ 0, 0, 0 })
{}

void Shape::setMaterial(std::shared_ptr<Material> material)
{
	mMaterial = material;
}

void Shape::setColour(Colour const& col)
{
	mColour = col;
}

Colour Shape::getColour() const
{
	return mColour;
}

Matte::Matte(Colour diffuseColour) : Material(diffuseColour) {}

Colour Matte::shade(Ray& const ray, SurfaceData& const surfaceData)
{
	Vector wo = -ray.o;
	Colour L = surfaceData.ambient->getColour() * diffuseColour;
	size_t numLights = surfaceData.lights->size();

	for (size_t i{ 0 }; i < numLights; ++i)
	{
		Vector wi = (*surfaceData.lights)[i]->getDirection(surfaceData);
		float nDotWi = glm::dot(surfaceData.normal, wi);

		if (nDotWi > 0.0f)
		{
			L += diffuseColour * glm::one_over_pi<float>() *
				(*surfaceData.lights)[i]->getColour() * (*surfaceData.lights)[i]->getRadiance(surfaceData) *
				nDotWi;
			L += specular(-glm::normalize(wi), -ray.d, surfaceData);
		}
	}
	L.x = glm::clamp<float>(L.x, 0, 1);
	L.y = glm::clamp<float>(L.y, 0, 1);
	L.z = glm::clamp<float>(L.z, 0, 1);

	return L;
}

Colour Matte::specular(Vector wi, Vector camDir, SurfaceData& const surfaceData)
{
	Vector r = glm::reflect(wi, surfaceData.normal);

	float vDotr = glm::dot(r, camDir);

	Colour L{ 0,0,0 };
	if (vDotr > 0.0)
	{
		L = Colour( specularCoefficient * glm::pow(vDotr, specularExp) );
	}

	return L;
}

// ***** Sphere function members *****
Sphere::Sphere(Point center, float radius) :
	mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(Ray const& ray, SurfaceData& data) const
{
	Vector temp = ray.o - mCentre;
	float a = glm::dot(ray.d, ray.d);
	float b = 2.0f * glm::dot(temp, ray.d);
	float c = glm::dot(temp, temp) - mRadiusSqr;
	float disc = b * b - 4.0f * a * c;

	if (disc < 0.0f)
	{
		return false;
	}

	double e = sqrt(disc);
	double denom = 2.0 * a;
	float t = (-b - e) / denom;    // smaller root

	if (t < data.t)
	{
		data.intersection = ray.o + (t * ray.d);
		data.normal = glm::normalize(data.intersection - mCentre);
		data.t = t;
		data.material = mMaterial;

		return true;
	}
	return false;
}


// ***** Plane function members *****
Plane::Plane(Point point, Vector normal) :
	point{ point }, normal{ normal }
{}


bool Plane::hit(Ray const& ray, SurfaceData& data) const
{
	float denom = glm::dot(normal, ray.d);
	if (abs(denom) < 0.000001f) return false;

	float t = glm::dot((point - ray.o), normal) / denom;
	if (t > 0)
	{
		if (t < data.t)
		{
			data.normal = glm::normalize(normal);
			data.t = t;
			data.material = mMaterial;
			data.intersection = ray.o + (ray.d * t);

			return true;
		}
	}

	return false;
}


// ***** Triangle function members *****
Triangle::Triangle(Point point1, Point point2, Point point3) :
	p0{ point1 }, p1{ point2 }, p2{ point3 }
{}

bool Triangle::hit(Ray const& ray, SurfaceData& data) const
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
		return false;
	}

	glm::vec3 qvec = glm::cross(rayv0, v1v0);

	float v = glm::dot(ray.d, qvec) * invDet;

	if (v < 0 || u + v > 1)
	{
		return false;
	}

	float t = glm::dot(v2v0, qvec) * invDet;

	if (t < data.t)
	{
		data.normal = glm::normalize(glm::cross(p0, p1));
		data.t = t;
		data.material = mMaterial;
		data.intersection = ray.o + (ray.d * t);

		return true;
	}
	return false;
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