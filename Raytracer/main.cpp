#include "Raytracer.hpp"


void populateGeometry(std::vector<std::shared_ptr<Shape>>& shapes)
{
	std::shared_ptr<Matte> matteMatRed = std::make_shared<Matte>(Matte{ Colour{1.0f, 0.4f, 0.2f} });

	std::shared_ptr<Matte> matteMatGreen = std::make_shared<Matte>(Matte{ Colour{0.2f, 1.0f, 0.2f} });

	std::shared_ptr<Matte> matteMatBlue = std::make_shared<Matte>(Matte{ Colour{0.2f, 0.4f, 1.0f} });

	for (size_t i{ 3 }; i < 7; i++)
	{
		Sphere s{ { -350 + (60.0f * i), 10.0f * i, -150.0f + 30.0f * i }, 10.0f + (10.0f * i) };
		s.setColour({ 0.15f, 0.1f * i, 0.15f });
		s.setMaterial(matteMatRed);
		shapes.push_back(std::make_shared<Sphere>(s));
	}
	
	Plane p0{ { 0, 0, -500 }, glm::normalize(Vector{ -1, 0, 1 }) };
	p0.setMaterial(matteMatGreen);
	p0.setColour({ 0.1f, 0.5f, 0.5f });

	//shapes.push_back(std::make_shared<Plane>(p0));

	Plane p1{ { 0, -50, 0 }, glm::normalize(Vector{ 0.0f, 1, 0.0f }) };
	p1.setMaterial(matteMatBlue);
	p1.setColour({ 0.5f, 0.1f, 0.5f });
	
	shapes.push_back(std::make_shared<Plane>(p1));
	

	Triangle t0{ { 400, -100, 200 },
				{ 600, -100, 200 },
				{ 500, 400, 200 } };
	t0.setColour({ 0.1f, 0.1f, 0.5f });
	t0.setMaterial(matteMatGreen);

	shapes.push_back(std::make_shared<Triangle>(t0));
	
	Triangle t1{ { 0, -100, -10 },
				{ 200, -100, -10 },
				{ 100, 400, -10 } };
	t1.setMaterial(matteMatGreen);

	shapes.push_back(std::make_shared<Triangle>(t1));
	
}

int main()
{
	std::shared_ptr<DirectionalLight> dirLight = std::make_shared<DirectionalLight>(DirectionalLight(Colour{ 1.0f, 1.0f, 1.0f }, glm::normalize(Vector{ 0.0f, 2.0f, 1.0f })));
	dirLight->setRadiance(1.5f);

	std::shared_ptr<PointLight> pointLight = std::make_shared<PointLight>(PointLight(Colour{ 1.0f, 1.0f, 1.0f }, Point{ 450.0f, 400.0f, 400.0f }));
	pointLight->setRadiance(0.003f);

	std::vector<std::shared_ptr<Light>> lights;
	//lights.push_back(dirLight);
	lights.push_back(pointLight);

	AmbientLight ambLight{ Colour{ 0.1f, 0.1f, 0.1f } };
	int ambientOccSamples = 8;

	FishEye camera; 
	camera.setEye({ 0,0,300 });
	camera.setLookAt({ 0,0,-100 });
	camera.setFrustrumDistance(150.0f);
	camera.computeUVW();

	std::vector<std::shared_ptr<Shape>> shapes;
	populateGeometry(shapes);

	std::vector<Colour> image(camera.width * camera.height);

	Vector rayDirTotal{ 0,0,0 };
	for (std::size_t y{ 0 }; y < camera.height; ++y)
	{
		for (std::size_t x{ 0 }; x < camera.width; ++x)
		{
			Colour pixel{ 0, 0, 0 };
			
			// Jittered sampling
			for (std::size_t py{ 0 }; py < 4; py++)
			{
				for (std::size_t px{ 0 }; px < 4; px++)
				{
					SurfaceData sd;
					sd.ambient = std::make_shared<Light>(ambLight);
					sd.lights = std::make_shared<std::vector<std::shared_ptr<Light>>>(lights);
					// Compute origin of ray. 
					// -0.5 to get the corner of each pixel then go through each grid corner and add random range of grid section size (1 / 4)
					float originX = (x - 0.5f * (camera.width - 1.0f)) - 0.5f + (px * 0.25f) + randomRange(0, 0.25f);
					float originY = (y - 0.5f * (camera.height - 1.0f)) - 0.5f + (py * 0.25f) + randomRange(0, 0.25f);
					
					Ray ray = camera.calculateRay(originX, originY);

					bool hit = false;
					//std::cout << ray.d.x << " : " << ray.d.y << " : " << ray.d.z << std::endl;
					for (std::size_t i{ 0 }; i < shapes.size(); i++)
					{
						if (shapes[i]->hit(ray, sd))
						{
							hit = true;
						}
					}
					if (hit)
					{
						Ray shadowRay;
						size_t numLights = sd.lights->size();

						SurfaceData temp;
						// Shadows
						for (std::size_t i{ 0 }; i < shapes.size(); i++)
						{
							for (size_t k{ 0 }; k < numLights; ++k)
							{
								temp.t = glm::distance(sd.intersection, (*sd.lights)[k]->getPoint(sd)) - 0.01f;

								shadowRay.d = -glm::normalize((*sd.lights)[k]->getDirection(sd));
								shadowRay.o = (*sd.lights)[k]->getPoint(sd);

								if (shapes[i]->hit(shadowRay, temp))
								{
									sd.shadowed = true;
									break;
								}
							}
						}
						pixel += sd.material->shade(ray, sd); 
					}
					
				}
			}

			// Average
			pixel.r /= 16.0f;
			pixel.g /= 16.0f;
			pixel.b /= 16.0f;
			
			image[x + y * camera.height] = pixel;
		}
		std::cout << y << " : " << std::endl;
	}

	saveToFile("sphere.bmp", camera.width, camera.height, image);
	return 0;
}

// Light
Light::Light(Colour colour) : colour(colour), radiance(1) {}
Vector Light::getDirection(SurfaceData& const sd)
{
	return { 0,0,0 };
}
float Light::getRadiance(SurfaceData& const sd)
{
	return radiance;
}
Point Light::getPoint(SurfaceData& const sd)
{
	return { 0,0,0 };
}
void Light::setRadiance(float radiance)
{
	this->radiance = radiance;
}
Colour Light::getColour()
{
	return colour;
}

// PointLight
PointLight::PointLight(Colour colour, Point point) : Light(colour), point(point) {}
float PointLight::getRadiance(SurfaceData& const sd)
{
	return radiance * glm::clamp(glm::distance(sd.intersection, point), 0.0f, 1.0f);
}
Vector PointLight::getDirection(SurfaceData& const sd)
{
	return point - sd.intersection;
}
Point PointLight::getPoint(SurfaceData& const sd)
{
	return point;
}

// AmbientLight
AmbientLight::AmbientLight(Colour colour) : Light(colour) {}

Vector AmbientLight::getDirection(SurfaceData& const sd)
{
	return { 0,0,0 };
}

// DirectionalLight
DirectionalLight::DirectionalLight(Colour colour, Vector direction) : Light(colour), direction(direction) {}

Vector DirectionalLight::getDirection(SurfaceData& const sd)
{	
	return direction;
}

Point DirectionalLight::getPoint(SurfaceData& const sd)
{
	return -direction * FLT_MAX;
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

Ray Camera::calculateRay(float x, float y)
{
	Ray ray;
	ray.o = getmEye();
	ray.d = glm::normalize((static_cast<float>(x)* getmV()) + (static_cast<float>(y)* getmU() - (getFrustrumDistance() * getmW())));

	return ray;
}

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
	mW = glm::normalize(mEye - mLookAt);
	mV = glm::normalize(glm::cross(mW, mUp));
	mU = glm::normalize(glm::cross(mW, mV));
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

Ray FishEye::calculateRay(float x, float y)
{
	Ray ray;
	ray.o = getmEye();
	ray.d = { 0,0,0 };

	float xn = (2.0f / width) * x;
	float yn = (2.0f / height) * y;
	float r_squared = xn * xn + yn * yn;
	//std::cout << xn << " : " << yn << std::endl;
	if (r_squared <= 1.0f) {
		float r = glm::sqrt(r_squared);
		float psi = r * psi_max * 0.0174532925199432957f / 2;
		float sin_psi = glm::sin(psi);
		float cos_psi = glm::cos(psi);
		float sin_alpha = yn / r;
		float cos_alpha = xn / r;
		Vector dir = (sin_psi * cos_alpha * mV) + 
					(sin_psi * sin_alpha * mU) -
					(cos_psi * mW);
		ray.d = dir;
		//std::cout << mU.x << " : " << mU.y << " : " << mU.z << std::endl;
	}

	return ray;
}

void FishEye::setFOV(float fov)
{
	psi_max = fov;
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

	if (!surfaceData.shadowed)
	{
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
void raytrace(int beginBlock, int endBlock, Camera const& camera, std::shared_ptr<World> world)
{
    Point samplePoint{}, pixelPoint{};
    atlas::math::Ray<atlas::math::Vector> ray{ {0, 0, 0}, {0, 0, -1} };

    float avg{ 1.0f / world->sampler->getNumSamples() };

    for (int r{ beginBlock }; r < endBlock; ++r)
    {
        for (int c{ 0 }; c < world->width; ++c)
        {
            Colour pixelAverage{ 0, 0, 0 };

            for (int j = 0; j < world->sampler->getNumSamples(); ++j)
            {
                ShadeRec trace_data{};
                trace_data.world = world;
                trace_data.t = std::numeric_limits<float>::max();
                samplePoint = world->sampler->sampleUnitSquare();
                pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
                pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
                camera.calculateRay(pixelPoint.x, pixelPoint.y, ray);

                bool hit{};

                for (auto obj : world->scene)
                {
                    hit |= obj->hit(ray, trace_data);
                }

                if (hit)
                {
                    pixelAverage += trace_data.material->shade(trace_data);
                }
            }

            world->image[(r * world->height) + c] = {   pixelAverage.r * avg,
                                                        pixelAverage.g * avg,
                                                        pixelAverage.b * avg };

        }
        std::cout << r << std::endl;
    }
}

int main()
{
    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    std::shared_ptr<World> world{ std::make_shared<World>() };

    world->width = 1000;
    world->height = 1000;
    world->background = { 0, 0, 0 };
    world->sampler = std::make_shared<Random>(4, 83);

    world->scene.push_back(
        std::make_shared<Triangle>(Triangle{ Point{-300, 0, -300}, Point{300, 0, -300}, Point{0, 300, -300} }));
    world->scene[0]->setMaterial(
        std::make_shared<Matte>(0.50f, 0.05f, Colour{ 1, 0, 0 }));
    world->scene[0]->setColour({ 1, 0, 0 });

    std::optional<atlas::utils::ObjMesh> optObjMesh = atlas::utils::loadObjMesh(modelRoot + "/teapot/teapot.obj");
    world->scene.push_back(
        std::make_shared<Mesh>(Mesh{ optObjMesh.value(), "teapot" }));

    world->ambient = std::make_shared<Ambient>();
    world->lights.push_back(
        std::make_shared<Directional>(Directional{ {0, 0, 1024} }));

    world->ambient->setColour({ 1, 1, 1 });
    world->ambient->scaleRadiance(0.05f);

    world->lights[0]->setColour({ 1, 1, 1 });
    world->lights[0]->scaleRadiance(4.0f);
    
    int numThreads = std::thread::hardware_concurrency();
    std::vector <std::shared_ptr<std::thread>> threads;

    Camera camera{ Point{0,0,200}, Point{0,0,-100}, Vector{0,1,0}, 100.0f };

    world->image = std::vector<Colour>(world->height * world->width, Colour{ 0,0,0 });

    for (size_t i{ 0 }; i < numThreads; i++)
    {
        int columnUnitSize = world->width / numThreads;
        int beginX = i * columnUnitSize;
        int endX = beginX + columnUnitSize;
        threads.push_back(std::make_shared<std::thread>(raytrace, beginX, endX, camera, world));
    }

    for (size_t i{ 0 }; i < threads.size(); i++)
    {
        threads[i]->join();
    }
    /*
    Point samplePoint{}, pixelPoint{};
    Ray<atlas::math::Vector> ray{ {0, 0, 0}, {0, 0, -1} };

    float avg{ 1.0f / world->sampler->getNumSamples() };

    for (int r{ 0 }; r < world->height; ++r)
    {
        for (int c{ 0 }; c < world->width; ++c)
        {
            Colour pixelAverage{ 0, 0, 0 };

            for (int j = 0; j < world->sampler->getNumSamples(); ++j)
            {
                ShadeRec trace_data{};
                trace_data.world = world;
                trace_data.t = std::numeric_limits<float>::max();
                samplePoint = world->sampler->sampleUnitSquare();
                pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
                pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
                camera.calculateRay(pixelPoint.x, pixelPoint.y, ray);

                bool hit{};

                for (auto obj : world->scene)
                {
                    hit |= obj->hit(ray, trace_data);
                }

                if (hit)
                {
                    pixelAverage += trace_data.material->shade(trace_data);
                }
            }

            world->image.push_back({ pixelAverage.r * avg,
                                    pixelAverage.g * avg,
                                    pixelAverage.b * avg });

        }
        std::cout << r << std::endl;
    */

    saveToFile("raytrace.bmp", world->width, world->height, world->image);

    return 0;
}

float randomRange(float min, float max)
{
	return (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + min) * max;
}