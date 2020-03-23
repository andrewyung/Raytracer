#include "Raytracer.hpp"

// ******* Function Member Implementation *******

// ***** ThreadPool function members *****    

ThreadPool::ThreadPool(unsigned int const& numThreads)
    : mTerminatePool(false)
{
    for (size_t i{ 0 }; i < numThreads; i++)
    {
        mThreads.push_back(std::thread(&ThreadPool::loopFunction, this));
    }
}

void ThreadPool::loopFunction()
{
    while (true)
    {
        std::function<void()> Job;

        {
            std::unique_lock<std::mutex> lock(mQueueMutex);

            mCondition.wait(lock, [&] {return !mQueue.empty() || mTerminatePool; });
            if (mTerminatePool && mQueue.empty()) break;
            Job = mQueue.front();
            mQueue.pop();
        }
        mJobsRunning++;
        Job(); 
        if (--mJobsRunning == 0 && mQueue.empty()) {
            mComplCondition.notify_one();
        }
    }
}

void ThreadPool::startJob(std::function<void()> job)
{
    {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        mQueue.push(job);
    }
    mCondition.notify_one();
}

void ThreadPool::waitDone()
{
    std::unique_lock<std::mutex> lock(mComplMutex);
    mComplCondition.wait(lock);
}

ThreadPool::~ThreadPool()
{
    // Get mutex to set terminatePool so all threads can be woken
    {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        mTerminatePool = true;
    }

    mCondition.notify_all();

    // Join all threads.
    for (std::thread& thread : mThreads)
    {
        thread.join();
    }

    mThreads.clear();
}

// ***** ImageTexture function members ******

ImageTexture::ImageTexture(std::string const& imageFilePath)
    : mWidth(0), mHeight(0), mChannels(0)
{
    mImage = stbi_load(imageFilePath.c_str(),
        &mWidth,
        &mHeight,
        &mChannels,
        STBI_rgb_alpha);


    if (mImage)
    {
        mSet = true;
    }
    else
    {
        mSet = false;
    }
}

ImageTexture::~ImageTexture()
{
    mSet = false;
    stbi_image_free(mImage);
}

ColourAlpha ImageTexture::getColour(float u, float v)
{
    if (mSet)
    {
        int x = floor(u * mWidth);
        int y = floor(v * mHeight);

        unsigned bytePerPixel = mChannels;
        unsigned char* pixelOffset = mImage + (x + (mHeight * y)) * mChannels;
        unsigned char r = pixelOffset[0];
        unsigned char g = pixelOffset[1];
        unsigned char b = pixelOffset[2];
        unsigned char a = mChannels >= 4 ? pixelOffset[3] : 0xff;
        return { r, g, b, a };
    }
    return { 0,0,0,0 };
}

// ***** BoundingVolumeBox *****
BoundingVolumeBox::BoundingVolumeBox(Vector const& startPoint)
    : mXLeftPlane(Plane{ startPoint, {1,0,0} }), mXRightPlane(Plane{ startPoint, { 1,0,0 } }),
      mYLeftPlane(Plane{ startPoint, {0,1,0} }), mYRightPlane(Plane{ startPoint, { 0,1,0 } }),
      mZLeftPlane(Plane{ startPoint, {0,0,1} }), mZRightPlane(Plane{ startPoint, { 0,0,1 } })
{}

void BoundingVolumeBox::addVolumePoint(Point const& point)
{
    if (point.x < mXLeftPlane.getPoint().x)
    {
        mXLeftPlane.setPoint(point);
    }
    if (point.x > mXRightPlane.getPoint().x)
    {
        mXRightPlane.setPoint(point);
    }

    if (point.y < mYLeftPlane.getPoint().y)
    {
        mYLeftPlane.setPoint(point);
    }
    if (point.y > mYRightPlane.getPoint().y)
    {
        mYRightPlane.setPoint(point);
    }

    if (point.z < mZLeftPlane.getPoint().z)
    {
        mZLeftPlane.setPoint(point);
    }
    if (point.z > mZRightPlane.getPoint().z)
    {
        mZRightPlane.setPoint(point);
    }
}

bool BoundingVolumeBox::intersects(atlas::math::Ray<Vector> const& ray)
{
    float xMin = mXLeftPlane.getPoint().x;
    float xMax = mXRightPlane.getPoint().x;

    float yMin = mYLeftPlane.getPoint().y;
    float yMax = mYRightPlane.getPoint().y;

    float zMin = mZLeftPlane.getPoint().z;
    float zMax = mZRightPlane.getPoint().z;
    ShadeRec sr;
    Point hitPoint;

    // Checks if ray hits any surface of cube
    sr.t = std::numeric_limits<float>::max();
    if (mXLeftPlane.hit(ray, sr))
    {
        hitPoint = ray.o + (ray.d * sr.t);
        // Check if within bounds of cube face
        if (hitPoint.y > yMin && hitPoint.y < yMax &&
            hitPoint.z > zMin && hitPoint.z < zMax) return true;
    }
    sr.t = std::numeric_limits<float>::max();
    if (mXRightPlane.hit(ray, sr))
    {
        hitPoint = ray.o + (ray.d * sr.t);
        if (hitPoint.y > yMin && hitPoint.y < yMax &&
            hitPoint.z > zMin && hitPoint.z < zMax) return true;
    }
    sr.t = std::numeric_limits<float>::max();
    if (mYLeftPlane.hit(ray, sr))
    {
        hitPoint = ray.o + (ray.d * sr.t);
        if (hitPoint.x > xMin && hitPoint.x < xMax &&
            hitPoint.z > zMin && hitPoint.z < zMax) return true;
    }
    sr.t = std::numeric_limits<float>::max();
    if (mYRightPlane.hit(ray, sr))
    {
        hitPoint = ray.o + (ray.d * sr.t);
        if (hitPoint.x > xMin && hitPoint.x < xMax &&
            hitPoint.z > zMin && hitPoint.z < zMax) return true;
    }
    sr.t = std::numeric_limits<float>::max();
    if (mZLeftPlane.hit(ray, sr))
    {
        hitPoint = ray.o + (ray.d * sr.t);
        if (hitPoint.x > xMin && hitPoint.x < xMax &&
            hitPoint.y > yMin && hitPoint.y < yMax) return true;
    }
    sr.t = std::numeric_limits<float>::max();
    if (mZRightPlane.hit(ray, sr))
    {
        hitPoint = ray.o + (ray.d * sr.t);
        if (hitPoint.x > xMin && hitPoint.x < xMax &&
            hitPoint.y > yMin && hitPoint.y < yMax) return true;
    }

    return false;
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

void Shape::setMaterial(std::shared_ptr<Material> const& material)
{
    mMaterial = material;
}

std::shared_ptr<Material> Shape::getMaterial() const
{
    return mMaterial;
}

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
    mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
{
    mSamples.reserve(mNumSets* mNumSamples);
    setupShuffledIndeces();
}

int Sampler::getNumSamples() const
{
    return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
    mShuffledIndeces.reserve(mNumSamples * mNumSets);
    std::vector<int> indices;

    std::random_device d;
    std::mt19937 generator(d());

    for (int j = 0; j < mNumSamples; ++j)
    {
        indices.push_back(j);
    }

    for (int p = 0; p < mNumSets; ++p)
    {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (int j = 0; j < mNumSamples; ++j)
        {
            mShuffledIndeces.push_back(indices[j]);
        }
    }
}

atlas::math::Point Sampler::sampleUnitSquare()
{
    if (mCount % mNumSamples == 0)
    {
        atlas::math::Random<int> engine;
        mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
    }

    return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

// ***** Light function members *****
Colour Light::L([[maybe_unused]] ShadeRec& sr)
{
    return mRadiance * mColour;
}

void Light::scaleRadiance(float b)
{
    mRadiance = b;
}

void Light::setColour(Colour const& c)
{
    mColour = c;
}

// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :
    mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    atlas::math::Vector tmp = ray.o - mCentre;
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
        sr.normal = (tmp + t * ray.d) / mRadius;
        sr.ray = ray;
        sr.color = mColour;
        sr.t = t;
        sr.material = mMaterial;
    }

    return intersect;
}

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    const auto tmp{ ray.o - mCentre };
    const auto a{ glm::dot(ray.d, ray.d) };
    const auto b{ 2.0f * glm::dot(ray.d, tmp) };
    const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
    const auto disc{ (b * b) - (4.0f * a * c) };

    if (atlas::core::geq(disc, 0.0f))
    {
        const float kEpsilon{ 0.01f };
        const float e{ std::sqrt(disc) };
        const float denom{ 2.0f * a };

        // Look at the negative root first
        float t = (-b - e) / denom;
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }

        // Now the positive root
        t = (-b + e);
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }
    }

    return false;
}

// ***** Triangle function members *****

Triangle::Triangle(Point p0, Point p1, Point p2, Vector2 p0UV, Vector2 p1UV, Vector2 p2UV)
    : mV0(p0), mV1(p1), mV2(p2), mUV0(p0UV), mUV1(p1UV), mUV2(p2UV), mBarycenCoords{ 0,0,0 }
{}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    if (t < sr.t)
    {
        glm::vec3 v2v0 = mV2 - mV0;
        glm::vec3 v1v0 = mV1 - mV0;
        sr.normal = glm::normalize(glm::cross(v1v0, v2v0));
        sr.t = t;
        sr.ray = ray;
        sr.color = mColour;
        sr.material = mMaterial;
        sr.uCoord = (mUV0.x * mBarycenCoords.x) + (mUV1.x * mBarycenCoords.y) + (mUV2.x * mBarycenCoords.z);
        sr.vCoord = (mUV0.y * mBarycenCoords.x) + (mUV1.y * mBarycenCoords.y) + (mUV2.y * mBarycenCoords.z);

        return true;
    }
    return false;
}

Vector Triangle::getV0Point() const
{
    return mV0;
}
Vector Triangle::getV1Point() const
{
    return mV1;
}
Vector Triangle::getV2Point() const
{
    return mV2;
}

bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    glm::vec3 v2v0 = mV2 - mV0;
    glm::vec3 v1v0 = mV1 - mV0;
    glm::vec3 rayv0 = ray.o - mV0;
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
    tMin = t;

    mBarycenCoords = Vector{ u, v, 1 - u - v };

    return true;
}

// ***** Triangle function members *****
Mesh::Mesh(atlas::utils::ObjMesh const& mesh, std::string modelSubDirName)
{
    std::vector<std::shared_ptr<Textured>> loadedMaterials;

    for (size_t i{ 0 }; i < mesh.materials.size(); i++)
    {
        loadedMaterials.push_back(
            std::make_shared<Textured>(mesh.materials[i], modelSubDirName));
    }
    // Go through each shape
    for (size_t i{ 0 }; i < mesh.shapes.size(); i ++)
    {
        atlas::utils::Shape shape = mesh.shapes[i];

        // Check if number of indices are summable by 3
        if (shape.indices.size() % 3 != 0) return;

        // Go through each triangle in shape
        for (size_t k{ 0 }; k < mesh.shapes[i].indices.size() - 2; k += 3)
        {
            atlas::utils::Vertex v0 = shape.vertices[shape.indices[k]];
            atlas::utils::Vertex v1 = shape.vertices[shape.indices[k + 1]];
            atlas::utils::Vertex v2 = shape.vertices[shape.indices[k + 2]];
            mMeshTriangles.push_back(Triangle{ v0.position, v1.position, v2.position,
                                                v0.texCoord, v1.texCoord, v2.texCoord, });

            if (!shape.materialIds.empty())
            {
                mMeshTriangles[mMeshTriangles.size() - 1].setMaterial(
                    loadedMaterials[shape.materialIds[i]]);
            }
        }
    }

    mBoundVolumeBox = std::make_unique<BoundingVolumeBox>(mMeshTriangles[0].getV0Point());
    for (size_t i{ 0 }; i < mMeshTriangles.size(); i++)
    {
        mBoundVolumeBox->addVolumePoint(mMeshTriangles[i].getV0Point());
        mBoundVolumeBox->addVolumePoint(mMeshTriangles[i].getV1Point());
        mBoundVolumeBox->addVolumePoint(mMeshTriangles[i].getV2Point());
    }
}
bool Mesh::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };

    ShadeRec nearestSR;
    bool intersected = false;

    bool volumeIntersect = mBoundVolumeBox->intersects(ray);
    if (!volumeIntersect) return false;

    for (size_t i{ 0 }; i < mMeshTriangles.size(); i++)
    {
        bool intersect{ mMeshTriangles[i].hit(ray, sr) };

        if (intersect && sr.t < t)
        {
            t = sr.t;
            nearestSR = sr;
            intersected = true;
        }
    }

    return intersected;
}

bool Mesh::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    return false;
}

// ***** Plane function members *****
Plane::Plane(Point point, Vector normal) : mPoint(point), mNormal(normal)
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    if (t < sr.t)
    {
        sr.normal = mNormal;
        sr.t = t;
        sr.ray = ray;
        sr.color = mColour;
        sr.material = mMaterial;

        return true;
    }
    return false;
}

bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    float denom = glm::dot(mNormal, ray.d); 
    if (abs(denom) < 0.00001f) return false;

    float t = abs(glm::dot((mPoint - ray.o), mNormal) / denom);
    if (t < tMin)
    {
        tMin = t;
        return true;
    }

    return false;
}

void Plane::setPoint(Point const& point)
{
    mPoint = point;
}

const Vector& Plane::getPoint() const
{
    return mPoint;
}


// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Regular::generateSamples()
{
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

    for (int j = 0; j < mNumSets; ++j)
    {
        for (int p = 0; p < n; ++p)
        {
            for (int q = 0; q < n; ++q)
            {
                mSamples.push_back(
                    atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
            }
        }
    }
}

// ***** Regular function members *****
Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Random::generateSamples()
{
    atlas::math::Random<float> engine;
    for (int p = 0; p < mNumSets; ++p)
    {
        for (int q = 0; q < mNumSamples; ++q)
        {
            mSamples.push_back(atlas::math::Point{
                engine.getRandomOne(), engine.getRandomOne(), 0.0f });
        }
    }
}

// ***** Lambertian function members *****
Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
{}

Lambertian::Lambertian(Colour diffuseColor, float diffuseReflection) :
    mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
{}

Colour
Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected,
    [[maybe_unused]] atlas::math::Vector const& incoming) const
{
    return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
}

Colour
Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected) const
{
    return mDiffuseColour * mDiffuseReflection;
}

void Lambertian::setDiffuseReflection(float kd)
{
    mDiffuseReflection = kd;
}

void Lambertian::setDiffuseColour(Colour const& colour)
{
    mDiffuseColour = colour;
}

// ***** Matte function members *****
Matte::Matte() :
    Material{},
    mDiffuseBRDF{ std::make_shared<Lambertian>() },
    mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

Matte::Matte(float kd, float ka, Colour color) : Matte{}
{
    setDiffuseReflection(kd);
    setAmbientReflection(ka);
    setDiffuseColour(color);
}

void Matte::setDiffuseReflection(float k)
{
    mDiffuseBRDF->setDiffuseReflection(k);
}

void Matte::setAmbientReflection(float k)
{
    mAmbientBRDF->setDiffuseReflection(k);
}

void Matte::setDiffuseColour(Colour colour)
{
    mDiffuseBRDF->setDiffuseColour(colour);
    mAmbientBRDF->setDiffuseColour(colour);
}

Colour Matte::shade(ShadeRec& sr)
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = -sr.ray.o;
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; ++i)
    {
        Vector wi = sr.world->lights[i]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) *
                nDotWi;
        }
    }

    return L;
}

// ***** Textured function members ***** 
Textured::Textured(tinyobj::material_t const& material, std::string const& modelSubDirName) 
    : mTexture(ImageTexture{ modelRoot + modelSubDirName + "/" + material.diffuse_texname })
{}

Colour Textured::shade(ShadeRec& sr)
{
    ColourAlpha colourA = mTexture.getColour(sr.uCoord, sr.vCoord);
    return colourA;
}

// ***** Directional function members *****
Directional::Directional() : Light{}
{}

Directional::Directional(atlas::math::Vector const& d) : Light{}
{
    setDirection(d);
}

void Directional::setDirection(atlas::math::Vector const& d)
{
    mDirection = glm::normalize(d);
}

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr)
{
    return mDirection;
}

// ***** Ambient function members *****
Ambient::Ambient() : Light{}
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr)
{
    return atlas::math::Vector{ 0.0f };
}

// ***** Camera function members *****
Camera::Camera(Point position, Point lookAt, Vector up, float frustrumDist)
    : mEye{ position }, mLookAt{ lookAt - position }, mFrustrumDist(frustrumDist)
{
    mW = glm::normalize(position - lookAt);
    mV = glm::normalize(glm::cross(mW, up));
    mU = glm::normalize(glm::cross(mW, mV));
}

void Camera::calculateRay(float x, float y, atlas::math::Ray<Vector>& ray) const
{
    ray.o = mEye;
    ray.d = glm::normalize(static_cast<float>(x) * mV + static_cast<float>(y) * mU - mFrustrumDist * mW);
}

// ******* Driver Code *******

void raytrace(int beginBlockY, int endBlockY, int beginBlockX, int endBlockX, Camera const& camera, std::shared_ptr<World> world)
{
    Point samplePoint{}, pixelPoint{};
    atlas::math::Ray<atlas::math::Vector> ray{ {0, 0, 0}, {0, 0, -1} };

    float avg{ 1.0f / world->sampler->getNumSamples() };

    for (int r{ beginBlockY }; r < endBlockY; ++r)
    {
        for (int c{ beginBlockX }; c < endBlockX; ++c)
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
    }
    PrintThread{} <<    "y: " << beginBlockY << "-" << endBlockY <<
                        "x: " << beginBlockX << "-" << endBlockX << 
                        " on thread: " << std::this_thread::get_id() << std::endl;
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
    
    unsigned int numThreads = std::thread::hardware_concurrency();

    ThreadPool threadPool{ numThreads };

    Camera camera{ Point{0,0,200}, Point{0,0,-100}, Vector{0,1,0}, 600.0f };

    world->image = std::vector<Colour>(world->height * world->width, Colour{ 0,0,0 });

    unsigned int gridN = numThreads * 5;
    // Split image into grid and assign to thread
    for (size_t y{ 0 }; y < gridN; y++)
    {
        for (size_t x{ 0 }; x < gridN; x++)
        {
            int columnUnitSize = world->width / gridN;
            int rowUnitSize = world->height / gridN;

            int beginY = y * rowUnitSize;
            int endY = beginY + rowUnitSize;
            int beginX = x * columnUnitSize;
            int endX = beginX + columnUnitSize;

            threadPool.startJob([=, &camera, &world] { raytrace(beginY, endY, beginX, endX, camera, world); });
        }
    }

    threadPool.waitDone();

    /*
    Vector colour;
    for (size_t i{ 0 }; i < world->image.size(); i++)
    {
        colour += world->image[i];
    }
    */
    saveToFile("raytrace.bmp", world->width, world->height, world->image);

    return 0;
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