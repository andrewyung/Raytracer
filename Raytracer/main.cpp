#include "Raytracer.hpp"

// Weighted cosine hemisphere sampling
Vector sampleHemisphere(float inX, float inY)
{
    /*
    const float r = sqrt(inX);
    const float theta = 2 * glm::pi<float>() * inY;

    const float x = r * cos(theta);
    const float y = r * sin(theta);

    return Vector{ x, sqrt(glm::max(0.0f, 1 - inX)), y };
    */
    const float r = sqrt(1.0f - inX * inX);
    const float phi = 2 * glm::pi<float>() * inY;

    return Vector{ cos(phi) * r, inX, sin(phi) * r };
}

glm::mat4 rotationMatToSpace(Vector up)
{
    Vector w = normalize(up);
    // Using jittered up
    Vector u = normalize(cross(w, { 0.053f, 1.0f, 0.0332f }));
    Vector v = cross(w, u);

    glm::mat4 rotationMatrix(1.0f);
    rotationMatrix[0] = { u, 0 };
    rotationMatrix[1] = { w, 0 };
    rotationMatrix[2] = { v, 0 };

    return rotationMatrix;
}

// ******* Function Member Implementation *******

// ***** Material function members *****

float Material::shadowed(ShadeRec const& sr, std::shared_ptr<Light> const& light)
{
    //Shadow ray
    atlas::math::Ray<Vector> shadowRay;
    shadowRay.o = light->getSourcePoint(sr);
    Vector intersectPoint = sr.ray(sr.t);
    shadowRay.d = normalize(intersectPoint - shadowRay.o);
    float distToLight = glm::length(intersectPoint - shadowRay.o) - 0.01f;

    ShadeRec rec{};
    rec.world = sr.world;
    rec.shadowRay = true;

    bool shadowed{ false };
    bool transparentOnly{ true };
    for (auto obj : rec.world->scene)
    {
        rec.t = distToLight;
        bool hit = obj->hit(shadowRay, rec);

        if (hit && rec.t < distToLight)
        {
            transparentOnly &= rec.material->mTransparent;
            shadowed = true;
        }
    }
    if (shadowed)
    {
        if (transparentOnly)
        {
            return 0.8f;
        }
        return 0;
    }

    return 1;
}

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
    mChannels = 4;

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

ColourAlpha ImageTexture::getColour(Vector2 uv)
{
    if (mSet)
    {
        int x = floor(uv.x * (mWidth));
        int y = floor(uv.y * (mHeight));

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

    updateMinMaxCentroid();
}

void BoundingVolumeBox::updateMinMaxCentroid()
{
    xMin = mXLeftPlane.getPoint().x;
    xMax = mXRightPlane.getPoint().x;

    yMin = mYLeftPlane.getPoint().y;
    yMax = mYRightPlane.getPoint().y;

    zMin = mZLeftPlane.getPoint().z;
    zMax = mZRightPlane.getPoint().z;

    mCentroid = {   (xMin + xMax) / 2.0f,
                    (yMin + yMax) / 2.0f,
                    (zMin + zMax) / 2.0f, };
}

bool BoundingVolumeBox::intersects(atlas::math::Ray<Vector> const& ray)
{
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

Point BoundingVolumeBox::getCentroid()
{
    return mCentroid;
}

std::vector<Point> BoundingVolumeBox::getBoundingBoxPoints()
{
    return {    mXLeftPlane.getPoint(),
                mXRightPlane.getPoint(),
                mYLeftPlane.getPoint(),
                mYRightPlane.getPoint(),
                mZLeftPlane.getPoint(),
                mZRightPlane.getPoint(),
    };
}

// ***** BVHAccel function members *****
BVHAccel::BVHAccel()
    : mDirty(false), mGenerated(false)
{}

BVHAccel::BVHNode::BVHNode(BoundingVolumeBox bvb)
    : nodeBvb(bvb), accesed(false), leaf(false)
{}

void BVHAccel::addShape(std::vector<Point> boundingBoxPoints, std::shared_ptr<Shape> shape)
{
    if (boundingBoxPoints.size() != 6) return;

    BoundingVolumeBox nodeBoundBox(boundingBoxPoints[0]);
    for (size_t i{ 1 }; i < boundingBoxPoints.size(); i++)
    {
        nodeBoundBox.addVolumePoint(boundingBoxPoints[i]);
    }

    BVHNode node(nodeBoundBox);

    node.leaf = true;
    node.shape = shape;

    mHeirarchy.push_back(node);
}

void BVHAccel::generateBVH()
{
    int leaves = mHeirarchy.size();
    int nextLayerLeaves = ceil(leaves / 2.0f);

    // Find nearest extents
    while (leaves != 1)
    {
        int layerEndRange = mHeirarchy.size();
        int layerStartRange = mHeirarchy.size() - leaves;
        for (size_t y{ 0 }; y < nextLayerLeaves; y++)
        {
            // find closest bounds
            size_t leaf1Index{ 0 };
            size_t leaf2Index{ 0 };
            int i;
            int k;
            bool set = false;
            float minCentroidDist = std::numeric_limits<float>().max();
            for (i = { layerEndRange - 1 }; i > layerStartRange; i--)
            {
                if (mHeirarchy[i].accesed) continue;

                for (k = { i - 1 }; k >= layerStartRange; k--)
                {
                    if (k < 0 || k < layerStartRange) break;
                    if (mHeirarchy[k].accesed) continue;

                    float centroidDist = distance(mHeirarchy[i].nodeBvb.getCentroid(), mHeirarchy[k].nodeBvb.getCentroid());
                    if (centroidDist < minCentroidDist)
                    {
                        minCentroidDist = centroidDist;
                        leaf1Index = i;
                        leaf2Index = k;
                        set = true;
                    }
                }
            }
            // These is only 1 node left so we push it onto the next layer
            if (!set)
            {
                // Find unaccessed node
                for (i = { layerEndRange - 1 }; i >= layerStartRange; i--)
                {
                    if (i < 0 || i < layerStartRange) break;
                    if (!mHeirarchy[i].accesed)
                    {
                        mHeirarchy.push_back(mHeirarchy[i]);
                        mHeirarchy[i].accesed = true;
                        break;
                    }
                }
            }
            else
            {
                // Create node
                BoundingVolumeBox leaf1Box = mHeirarchy[leaf1Index].nodeBvb;
                BoundingVolumeBox leaf2Box = mHeirarchy[leaf2Index].nodeBvb;
                BoundingVolumeBox newBoundVolBox(leaf1Box.getBoundingBoxPoints()[0]);
                for (size_t k{ 1 }; k < leaf1Box.getBoundingBoxPoints().size(); k++)
                {
                    newBoundVolBox.addVolumePoint(leaf1Box.getBoundingBoxPoints()[k]);
                }
                for (size_t k{ 0 }; k < leaf2Box.getBoundingBoxPoints().size(); k++)
                {
                    newBoundVolBox.addVolumePoint(leaf2Box.getBoundingBoxPoints()[k]);
                }
                BVHNode combinedNode{ newBoundVolBox };
                combinedNode.leaf = false;
                combinedNode.left = leaf1Index;
                combinedNode.right = leaf2Index;
                mHeirarchy.push_back(combinedNode);

                // Set these nodes to not be processed when finding closest bounds
                mHeirarchy[leaf1Index].accesed = true;
                mHeirarchy[leaf2Index].accesed = true;
            }
        }

        leaves = nextLayerLeaves;
        nextLayerLeaves = ceil(leaves / 2.0f);
    }
}

bool BVHAccel::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    std::queue<BVHNode> queuedNodes;
    BVHNode root = mHeirarchy[mHeirarchy.size() - 1];
    if (root.leaf)
    {
        if (root.shape->hit(ray, sr))
        {
            // Ray intersects shape in leaf node
            return true;
        }
        return false;
    }

    if (root.nodeBvb.intersects(ray))
    {
        queuedNodes.push(mHeirarchy[root.left]);
        queuedNodes.push(mHeirarchy[root.right]);
    }

    bool intersects = false;
    while (!queuedNodes.empty())
    {
        BVHNode node = queuedNodes.front();
        if (node.leaf)
        {
            if (node.shape->hit(ray, sr))
            {
                // Ray intersects shape in leaf node
                intersects = true;
            }
        }
        else
        {
            if (node.nodeBvb.intersects(ray))
            {
                queuedNodes.push(mHeirarchy[node.left]);
                queuedNodes.push(mHeirarchy[node.right]);
            }
        }
        queuedNodes.pop();
    }

    return intersects;
}

bool BVHAccel::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
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
    mSamples.reserve(mNumSets * mNumSamples);
    setupShuffledIndices();
}

int Sampler::getNumSamples() const
{
    return mNumSamples;
}

void Sampler::setupShuffledIndices()
{
    mShuffledIndices.reserve(mNumSamples * mNumSets);
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
            mShuffledIndices.push_back(indices[j]);
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

    return mSamples[mJump + mShuffledIndices[mJump + mCount++ % mNumSamples]];
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

std::vector<Point> Sphere::getBoundingBoxPoints()
{
    BoundingVolumeBox sphereBvb(mCentre + Vector{ mRadius, mRadius, mRadius });
    sphereBvb.addVolumePoint(mCentre - Vector{ mRadius, mRadius, mRadius });
    return sphereBvb.getBoundingBoxPoints();
}

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

        return true;
    }

    return false;
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
    : mV0(p0), mV1(p1), mV2(p2), mUV0(p0UV), mUV1(p1UV), mUV2(p2UV)
{}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };
    Vector barycentricCoords;
    bool intersect{ intersectRay(ray, t, barycentricCoords) };

    if (intersect && t < sr.t)
    {
        glm::vec3 v2v0 = mV2 - mV0;
        glm::vec3 v1v0 = mV1 - mV0;
        sr.normal = glm::normalize(glm::cross(v1v0, v2v0));
        sr.t = t;
        sr.ray = ray;
        sr.color = mColour;
        sr.material = mMaterial;
        // barycentric coords ordered in a certain way from ray-triangle calculations
        sr.uvCoord = (mUV1 * barycentricCoords.x) + (mUV2 * barycentricCoords.y) + (mUV0 * barycentricCoords.z);

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
    glm::vec3 v0v1 = mV1 - mV0;
    glm::vec3 v0v2 = mV2 - mV0;
    glm::vec3 pvec = cross(ray.d, v0v2);
    float det = dot(v0v1, pvec);

    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < 0.000001f) return false;
    float invDet = 1.0f / det;

    glm::vec3 tvec = ray.o - mV0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    glm::vec3 qvec = cross(tvec, v0v1);
    float v = dot(ray.d, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(v0v2, qvec) * invDet;

    tMin = t;

    return true;
}

std::vector<Point> Triangle::getBoundingBoxPoints()
{
    BoundingVolumeBox triangleBvb(mV0);
    triangleBvb.addVolumePoint(mV1);
    triangleBvb.addVolumePoint(mV2);
    return triangleBvb.getBoundingBoxPoints();
}

bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin, Vector& barycentricCoords) const
{
    glm::vec3 v0v1 = mV1 - mV0;
    glm::vec3 v0v2 = mV2 - mV0;
    glm::vec3 pvec = cross(ray.d, v0v2);
    float det = dot(v0v1, pvec);

    // ray and triangle are parallel if det is close to 0
    if (det < 0.000001f) return false;
    float invDet = 1.0f / det;

    glm::vec3 tvec = ray.o - mV0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    glm::vec3 qvec = cross(tvec, v0v1);
    float v = dot(ray.d, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(v0v2, qvec) * invDet;

    tMin = t;
    barycentricCoords = Vector{ u, v, 1.0f - u - v };

    return true;
}

// ***** MultiMesh function members *****
MultiMesh::MultiMesh(atlas::utils::ObjMesh const& mesh, std::string modelSubDirName, Vector offset)
{
    std::vector<std::shared_ptr<Textured>> loadedMaterials;

    for (size_t i{ 0 }; i < mesh.materials.size(); i++)
    {
        loadedMaterials.push_back(
            std::make_shared<Textured>(mesh.materials[i], modelSubDirName));
    }
    // Go through each shape
    for (size_t i{ 0 }; i < mesh.shapes.size(); i++)
    {
        meshes.push_back(Mesh(mesh.shapes[i], i, loadedMaterials, modelSubDirName, offset));
    }

    mBoundVolumeBox = std::make_unique<BoundingVolumeBox>(meshes[0].getBoundingBoxPoints()[0]);
    for (size_t i{ 0 }; i < meshes.size(); i++)
    {
        std::vector<Point> boundingBoxPoints = meshes[i].getBoundingBoxPoints();
        for (size_t k{ 0 }; k < boundingBoxPoints.size(); k++)
        {
            mBoundVolumeBox->addVolumePoint(boundingBoxPoints[k]);
        }
    }
}

bool MultiMesh::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };

    bool intersected = false;

    bool volumeIntersect = mBoundVolumeBox->intersects(ray);
    if (!volumeIntersect) return false;

    for (size_t i{ 0 }; i < meshes.size(); i++)
    {
        bool intersect{ meshes[i].hit(ray, sr) };

        if (intersect && sr.t < t)
        {
            t = sr.t;
            intersected = true;
        }
    }

    return intersected;
}

bool MultiMesh::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    return false;
}

std::vector<Point> MultiMesh::getBoundingBoxPoints()
{
    return mBoundVolumeBox->getBoundingBoxPoints();
}


// ***** Mesh function members *****
Mesh::Mesh(atlas::utils::ObjMesh const& mesh, std::string modelSubDirName, Vector offset)
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
            v0.position += offset;
            atlas::utils::Vertex v1 = shape.vertices[shape.indices[k + 1]];
            v1.position += offset;
            atlas::utils::Vertex v2 = shape.vertices[shape.indices[k + 2]];
            v2.position += offset;
            mMeshTriangles.push_back(Triangle{ v0.position, v1.position, v2.position,
                                                v0.texCoord, v1.texCoord, v2.texCoord, });

            if (!shape.materialIds.empty() && shape.materialIds[i] != -1)
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

Mesh::Mesh(atlas::utils::Shape shape, unsigned int matIndex, const std::vector<std::shared_ptr<Textured>>& loadedMaterials, std::string modelSubDirName, Vector offset)
{
    // Check if number of indices are summable by 3
    if (shape.indices.size() % 3 != 0) return;

    // Go through each triangle in shape
    for (size_t k{ 0 }; k < shape.indices.size() - 2; k += 3)
    {
        atlas::utils::Vertex v0 = shape.vertices[shape.indices[k]];
        v0.position += offset;
        atlas::utils::Vertex v1 = shape.vertices[shape.indices[k + 1]];
        v1.position += offset;
        atlas::utils::Vertex v2 = shape.vertices[shape.indices[k + 2]];
        v2.position += offset;
        mMeshTriangles.push_back(Triangle{ v0.position, v1.position, v2.position,
                                            v0.texCoord, v1.texCoord, v2.texCoord, });

        if (!shape.materialIds.empty())
        {
            mMeshTriangles[mMeshTriangles.size() - 1].setMaterial(
                loadedMaterials[shape.materialIds[matIndex]]);
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

std::vector<Point> Mesh::getBoundingBoxPoints()
{
    return mBoundVolumeBox->getBoundingBoxPoints();
}


bool Mesh::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };

    bool intersected = false;

    bool volumeIntersect = mBoundVolumeBox->intersects(ray);
    if (!volumeIntersect) return false;

    for (size_t i{ 0 }; i < mMeshTriangles.size(); i++)
    {
        bool intersect{ mMeshTriangles[i].hit(ray, sr) };

        if (intersect && sr.t < t)
        {
            t = sr.t;
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

    if (intersect && t < sr.t)
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

    float t = (glm::dot((mPoint - ray.o), mNormal) / denom);
    if (t >= 0 && t < tMin)
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

// ***** Jitter function members *****
Jitter::Jitter(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Jitter::generateSamples()
{
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));
    float gridSize = 1.0f / n;
    for (int j = 0; j < mNumSets; ++j)
    {
        for (int p = 0; p < n; ++p)
        {
            for (int q = 0; q < n; ++q)
            {
                mSamples.push_back(
                    atlas::math::Point{ q * gridSize + (static_cast<float>(rand()) / RAND_MAX * gridSize), 
                                        p * gridSize + (static_cast<float>(rand()) / RAND_MAX * gridSize), 
                                        0.0f });
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
Lambertian::Lambertian() : mDiffuseColour{1,1,1}, mDiffuseReflection{0.5f}
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

// ***** SpecularReflection function members *****
SpecularReflection::SpecularReflection()
    : mSpecularColour({ 1,1,1 }), mSpecularCofficient(0.3f), mSpecularExp(16)
{}    

void SpecularReflection::setSpecularCoefficient(float kd)
{
    mSpecularCofficient = kd;
}

void SpecularReflection::setSpecularExp(int exp)
{
    mSpecularExp = exp;
}

void SpecularReflection::setSpecularColour(Colour const& colour)
{
    mSpecularColour = colour;
}

Colour SpecularReflection::fn(ShadeRec const& sr,
    atlas::math::Vector const& reflected,
    atlas::math::Vector const& incoming) const
{
    Vector r = glm::reflect(incoming, sr.normal);

    float vDotr = glm::dot(r, -reflected);

    Colour L{ 0,0,0 };
    if (vDotr > 0.0)
    {
        L = Colour(mSpecularCofficient * glm::pow(vDotr, mSpecularExp));
    }

    return L;
}

Colour SpecularReflection::rho(ShadeRec const& sr,
    atlas::math::Vector const& reflected) const
{
    return { 0,0,0 };
}

// ***** Matte function members *****
Matte::Matte() :
    Material{},
    mDiffuseBRDF{ std::make_shared<Lambertian>() },
    mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

Matte::Matte(float kd, float ka, Colour color)
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

    Vector wo = -sr.ray.d;
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

// ***** Specular function members *****

Specular::Specular() :
    Material{},
    mSpecularBRDF{ std::make_shared<SpecularReflection>() },
    mDiffuseBRDF{ std::make_shared<Lambertian>() },
    mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

Specular::Specular(float ks, float kd, float ka, Colour color) :
    Material{},
    mSpecularBRDF{ std::make_shared<SpecularReflection>() },
    mDiffuseBRDF{ std::make_shared<Lambertian>() },
    mAmbientBRDF{ std::make_shared<Lambertian>() }
{
    setDiffuseReflection(ks);
    setDiffuseReflection(kd);
    setAmbientReflection(ka);
    setDiffuseColour(color);
}

void Specular::setSpecularCoefficient(float k)
{
    mSpecularBRDF->setSpecularCoefficient(k);
}

void Specular::setDiffuseReflection(float k)
{
    mDiffuseBRDF->setDiffuseReflection(k);
}

void Specular::setAmbientReflection(float k)
{
    mAmbientBRDF->setDiffuseReflection(k);
}

void Specular::setDiffuseColour(Colour colour)
{
    mSpecularBRDF->setSpecularColour(colour);
    mDiffuseBRDF->setDiffuseColour(colour);
    mAmbientBRDF->setDiffuseColour(colour);
}

void Specular::setSpecularExp(int k)
{
    mSpecularBRDF->setSpecularExp(k);
}

Colour Specular::shade(ShadeRec& sr)
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = normalize(-sr.ray.d);
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
    
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; ++i)
    {
        float shadowAttenuation = shadowed(sr, sr.world->lights[i]);

        Vector wi = normalize(sr.world->lights[i]->getDirection(sr));
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) *
               nDotWi * shadowAttenuation;
            L += mSpecularBRDF->fn(sr, wo, wi) * shadowAttenuation;
        }
    }

    L.x = glm::clamp<float>(L.x, 0, 1);
    L.y = glm::clamp<float>(L.y, 0, 1);
    L.z = glm::clamp<float>(L.z, 0, 1);
    return L;
}


// ***** Textured function members ***** 
Textured::Textured(tinyobj::material_t const& material, std::string const& modelSubDirName) 
    :   mTexture(ImageTexture{ modelRoot + modelSubDirName + "/" + material.diffuse_texname }),
        mAmbientBRDF{ std::make_shared<Lambertian>()},
        mSpecularBRDF{ std::make_shared<SpecularReflection>()}
{}

void Textured::setSpecularReflection(float ks)
{
    mSpecularBRDF->setSpecularCoefficient(ks);
}

void Textured::setSpecularExp(int exp)
{
    mSpecularBRDF->setSpecularExp(exp);
}

void Textured::setSpecularColour(Colour col)
{
    mSpecularBRDF->setSpecularColour(col);
}

Colour Textured::shade(ShadeRec& sr)
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    Colour texColour = (mTexture.getColour(sr.uvCoord) / 255.0f).xyz;
    Vector wo = normalize(-sr.ray.d);
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr) * texColour;
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; ++i)
    {
        float shadowAttenuation = shadowed(sr, sr.world->lights[i]);

        Vector wi = normalize(sr.world->lights[i]->getDirection(sr));
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            float lightToSurfaceDist = glm::length(sr.world->lights[i]->getSourcePoint(sr) - sr.ray(sr.t));
            float attenuation = lightAttenuationFactor / (lightToSurfaceDist * lightToSurfaceDist);
            L += texColour * sr.world->lights[i]->L(sr) *
                nDotWi * attenuation * shadowAttenuation;
            L += mSpecularBRDF->fn(sr, wo, wi) * attenuation * shadowAttenuation;
        }
    }
    L.x = glm::clamp<float>(L.x, 0, 1);
    L.y = glm::clamp<float>(L.y, 0, 1);
    L.z = glm::clamp<float>(L.z, 0, 1);
    return L;
}

void Textured::setAmbientReflection(float ka)
{
    mAmbientBRDF->setDiffuseReflection(ka);
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

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec const& sr)
{
    return mDirection;
}
atlas::math::Vector Directional::getSourcePoint([[maybe_unused]] ShadeRec const& sr)
{
    return sr.ray(sr.t) + (1000000.0f * mDirection);
}

// ***** PointLight function members *****
PointLight::PointLight()
    :   mPoint({ 0,0,0 })
{}
PointLight::PointLight(Point const& p)
    :   mPoint(p)
{}

atlas::math::Vector PointLight::getDirection([[maybe_unused]] ShadeRec const& sr)
{
    return mPoint - sr.ray(sr.t);
}

atlas::math::Vector PointLight::getSourcePoint([[maybe_unused]] ShadeRec const& sr)
{
    return mPoint;
}

// ***** Ambient function members *****
Ambient::Ambient() : Light{}
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec const& sr)
{
    return atlas::math::Vector{ 0.0f };
}

atlas::math::Vector Ambient::getSourcePoint([[maybe_unused]] ShadeRec const& sr)
{
    return atlas::math::Vector{ 0.0f };
}

Colour Ambient::L(ShadeRec& sr)
{
    glm::mat4 rotationMatrix = rotationMatToSpace(sr.normal);

    float gridBlockSize = 1.0f / mOcclusionSamples;

    float occlusionMaxSumDist = (mOcclusionRayDist * mOcclusionSamples * mOcclusionSamples);
    float occludedDistance = occlusionMaxSumDist;

    // Jittered
    for (size_t x{ 0 }; x < mOcclusionSamples; x++)
    {
        for (size_t y{ 0 }; y < mOcclusionSamples; y++)
        {
            float sampleX = static_cast<float>(x) * gridBlockSize + (((double)rand() / RAND_MAX) * gridBlockSize);
            float sampleY = static_cast<float>(y) * gridBlockSize + (((double)rand() / RAND_MAX) * gridBlockSize);
            atlas::math::Ray<Vector> reflectedRay;
            Vector r = sampleHemisphere(sampleX, sampleY);
            r = rotationMatrix * glm::vec4{ r, 0 };
            r = normalize(r);
            reflectedRay.d = r;
            reflectedRay.o = sr.ray(sr.t) + (0.01f * sr.normal);

            ShadeRec rec{};
            rec.world = sr.world;
            rec.depth = sr.depth + 1;
            rec.t = mOcclusionRayDist;

            bool hit{false};
            for (auto obj : rec.world->scene)
            {
                hit |= obj->hit(reflectedRay, rec);
            }

            if (hit)
            {
                occludedDistance -= (mOcclusionRayDist - rec.t);
            }
        }
    }

    float occlusionFactor = (occludedDistance / occlusionMaxSumDist);
    return mRadiance * mColour * occlusionFactor * occlusionFactor;
}

// ***** Camera function members *****
Camera::Camera(Point position, Point lookAt, Vector up, float frustrumDist)
    : mEye{ position }, mFrustrumDist(frustrumDist)
{
    mW = glm::normalize(position - lookAt);
    mV = glm::normalize(glm::cross(mW, up));
    mU = glm::normalize(glm::cross(mW, mV));
}

Point Camera::getEye() const
{
    return mEye;
}

void Camera::calculateRay(float x, float y, atlas::math::Ray<Vector>& ray) const
{
    ray.o = mEye;
    ray.d = glm::normalize(x * mV + y * mU - mFrustrumDist * mW);
}

// ******* Reflective function members *******
Reflective::Reflective()
{}

Colour Reflective::fn(ShadeRec const& sr,
    atlas::math::Vector const& reflected,
    atlas::math::Vector const& incoming) const
{
    return { 0,0,0 };
}

Colour Reflective::rho(ShadeRec const& sr,
    atlas::math::Vector const& reflected) const
{
    if (sr.depth > maxBounceDepth) return sr.color;

    if (sr.shadowRay) return { 0,0,0 };

    atlas::math::Ray<Vector> reflectedRay;
    Vector r = glm::reflect(-reflected, sr.normal);
    reflectedRay.d = r;
    reflectedRay.o = sr.ray.o + (sr.t * sr.ray.d) + (0.1f * r);

    ShadeRec rec{};
    rec.world = sr.world;
    rec.depth = sr.depth + 1; 
    rec.t = std::numeric_limits<float>::max();

    bool hit{};

    rec.color = { 0,0,0 };
    for (auto obj : rec.world->scene)
    {
        hit |= obj->hit(reflectedRay, rec);
    }

    if (hit)
    {
        rec.color = rec.material->shade(rec);
    }

    return rec.color;
}

// ******* Mirror function members *******
Mirror::Mirror() :
    Material{},
    mReflectiveBRDF{ std::make_shared<Reflective>() },
    mSpecularBRDF{ std::make_shared<SpecularReflection>() },
    mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

void Mirror::setAmbientReflection(float ka)
{
    mAmbientBRDF->setDiffuseReflection(ka);
}
void Mirror::setAmbientColour(Colour colour)
{
    mAmbientBRDF->setDiffuseColour(colour);
}

Colour Mirror::shade(ShadeRec& sr)
{
    Vector wo = normalize(-sr.ray.d);
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
   
    L += mReflectiveBRDF->rho(sr, wo);
    
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; ++i)
    {
        Vector wi = normalize(sr.world->lights[i]->getDirection(sr));
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            L += mSpecularBRDF->fn(sr, wo, wi);
        }
    }

    L.x = glm::clamp<float>(L.x, 0, 1);
    L.y = glm::clamp<float>(L.y, 0, 1);
    L.z = glm::clamp<float>(L.z, 0, 1);
    return L;
}

// ****** Transparency function members ******
Transparency::Transparency(float instalRefraction)
    :   mIndexOfRefraction{ instalRefraction }
{}

Colour Transparency::fn(ShadeRec const& sr,
    atlas::math::Vector const& reflected,
    atlas::math::Vector const& incoming) const
{
    return { 0,0,0 };
}

Colour Transparency::rho(ShadeRec const& sr,
    atlas::math::Vector const& reflected) const
{
    if (sr.depth > maxBounceDepth) return sr.color;

    if (sr.shadowRay) return { 0,0,0 };

    ShadeRec rec{};

    atlas::math::Ray<Vector> refractedRay;
    float enterIndexOfRefract = mIndexOfRefraction;
    Vector r = refraction(sr, enterIndexOfRefract);
    rec.indexOfRefraction = enterIndexOfRefract;
    refractedRay.d = r;
    refractedRay.o = sr.ray.o + (sr.t * sr.ray.d) + (0.1f * r);

    rec.world = sr.world;
    rec.depth = sr.depth + 1;
    rec.t = std::numeric_limits<float>::max();

    bool hit{};

    rec.color = { 0,0,0 };
    for (auto obj : rec.world->scene)
    {
        hit |= obj->hit(refractedRay, rec);
    }

    if (hit)
    {
        rec.color = rec.material->shade(rec);
    }

    return rec.color;
}

void Transparency::setIndexOfRefraction(float index)
{
    mIndexOfRefraction = index;
}

Vector Transparency::refraction(ShadeRec const& sr, float& enterIndexOfRefract) const
{
    float cosi = glm::clamp<float>(dot(sr.ray.d, sr.normal), -1.0f, 1.0f);

    Vector n = sr.normal;

    // Is ray going into or out of volume
    if (cosi < 0) 
    { 
        cosi = -cosi; 
    }
    else 
    { 
        n = -sr.normal; 
    }
    float eta = sr.indexOfRefraction / enterIndexOfRefract;
    float k = 1 - eta * eta * (1 - cosi * cosi);

    // Is total internal reflection
    if (k < 0)
    {
        enterIndexOfRefract = sr.indexOfRefraction;
        return glm::reflect(sr.ray.d, n);
    }

    return normalize(eta * sr.ray.d + (eta * cosi - sqrtf(k)) * n);
}

// ****** SemiTransparent function members ******

Refraction::Refraction()
    : mTransparencyBRDF{ std::make_shared<Transparency>() }
{
    mTransparent = true;
}

void Refraction::setIndexOfRefraction(float index)
{
    mTransparencyBRDF->setIndexOfRefraction(index);
}

Colour Refraction::shade(ShadeRec& sr)
{
    Vector wo = normalize(-sr.ray.d);
    Colour L = mTransparencyBRDF->rho(sr, wo);

    L.x = glm::clamp<float>(L.x, 0, 1);
    L.y = glm::clamp<float>(L.y, 0, 1);
    L.z = glm::clamp<float>(L.z, 0, 1);
    return L;
}

// ******* Driver Code *******

void raytrace(int beginBlockY, int endBlockY, int beginBlockX, int endBlockX, Camera const& camera, std::shared_ptr<World> const& world)
{
    Point samplePoint{}, pixelPoint{};
    atlas::math::Ray<atlas::math::Vector> ray;

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
                pixelPoint.x = (c - (0.5f * world->width)) + samplePoint.x;
                pixelPoint.y = (r - (0.5f * world->height)) + samplePoint.y;
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
    using atlas::core::Timer;
    Timer<float> timer; 
    timer.start();       
    float startTime = timer.elapsed();

    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    std::shared_ptr<World> world{ std::make_shared<World>() };

    world->width = 1000;
    world->height = 1000;
    world->background = { 0, 0, 0 };
    world->sampler = std::make_shared<Jitter>(4, 83);

    std::shared_ptr<BVHAccel> bvhAccel = std::make_shared<BVHAccel>();

    std::optional<atlas::utils::ObjMesh> teapotObjMesh = atlas::utils::loadObjMesh(modelRoot + "/teapot/teapot.obj");

    std::shared_ptr<Textured> teapotTexMaterial = std::make_shared<Textured>(teapotObjMesh.value().materials[0], "teapot");
    teapotTexMaterial->setAmbientReflection(0.7f);

    std::shared_ptr<Specular> specular = std::make_shared <Specular>();
    specular->setDiffuseColour({ 0.9f, 0.1f, 0.4f });
    specular->setDiffuseReflection(2.0f);

    std::shared_ptr<Specular> planeSpecular1 = std::make_shared <Specular>();
    planeSpecular1->setDiffuseColour({ 1.0f, 0.1f, 0.1f });
    planeSpecular1->setAmbientReflection(1.0f);

    std::shared_ptr<Specular> planeSpecular2 = std::make_shared <Specular>();
    planeSpecular2->setDiffuseColour({ 0.1f, 0.1f, 1.0f });
    planeSpecular2->setAmbientReflection(1.0f);

    std::shared_ptr<Specular> planeSpecular3 = std::make_shared <Specular>();
    planeSpecular3->setDiffuseColour({ 0.85f, 0.85f, 0.85f });
    planeSpecular3->setAmbientReflection(1.0f);

    std::shared_ptr<Specular> planeSpecular4 = std::make_shared <Specular>();
    planeSpecular4->setDiffuseColour({ 0.1f, 1.0f, 0.1f });
    planeSpecular4->setAmbientReflection(1.0f);

    std::shared_ptr<Mirror> mirror = std::make_shared <Mirror>();
    mirror->setAmbientColour({ 0.753f, 0.753f, 0.753f });
    mirror->setAmbientReflection(0.0f);

    std::shared_ptr<Refraction> refract = std::make_shared <Refraction>();

    std::shared_ptr<Triangle> triangle1 = std::make_shared<Triangle>(   Triangle{ Point{-200, 0, -300}, Point{200, 0, -300}, Point{0, 400, -300}, 
                                                                        Vector2{0.0f,0.0f}, Vector2{1.0f,0.0f}, Vector2{0.5f,1.0f} });
    triangle1->setMaterial(mirror);

    std::shared_ptr<Triangle> triangle2 = std::make_shared<Triangle>(Triangle{ Point{200, 0, 0}, Point{200, 0, 400}, Point{200, 400, 200},
                                                                        Vector2{0.0f,0.0f}, Vector2{1.0f,0.0f}, Vector2{0.5f,1.0f} });
    triangle2->setMaterial(teapotTexMaterial);

    std::shared_ptr<Sphere> sphere1 = std::make_shared<Sphere>(Sphere{ {-450,0,-350}, 100 });
    sphere1->setMaterial(specular);

    std::shared_ptr<Sphere> sphere2 = std::make_shared<Sphere>(Sphere{ {-350,0,300}, 100 });
    sphere2->setMaterial(mirror);
    std::shared_ptr<Sphere> sphere3 = std::make_shared<Sphere>(Sphere{ {-250,0,80}, 100 });
    sphere3->setMaterial(mirror);

    std::shared_ptr<Sphere> sphere4 = std::make_shared<Sphere>(Sphere{ {-250, 180, 150}, 100 });
    sphere4->setMaterial(refract);

    std::shared_ptr<Sphere> sphere5 = std::make_shared<Sphere>(Sphere{ {-600,-50,-50}, 30 });
    sphere5->setMaterial(planeSpecular4);
    std::shared_ptr<Sphere> sphere6 = std::make_shared<Sphere>(Sphere{ {-620,-30,-90}, 50 });
    sphere6->setMaterial(planeSpecular1);
    std::shared_ptr<Sphere> sphere7 = std::make_shared<Sphere>(Sphere{ {-605,-20,-55}, 40 });
    sphere7->setMaterial(planeSpecular2);


    std::shared_ptr<Plane> plane1 = std::make_shared<Plane>(Plane{ {0, -100, 0}, {0,1,0} });
    plane1->setMaterial(planeSpecular3);
    std::shared_ptr<Plane> plane3 = std::make_shared<Plane>(Plane{ {0, 550, 0}, {0,-1,0} });
    plane3->setMaterial(planeSpecular3);
    std::shared_ptr<Plane> plane4 = std::make_shared<Plane>(Plane{ {-1050, 0, 0}, {1,0,0} });
    plane4->setMaterial(planeSpecular3);

    std::shared_ptr<Plane> plane2 = std::make_shared<Plane>(Plane{ {350, 0, 0}, {-1,0,0} });
    plane2->setMaterial(planeSpecular4);
    std::shared_ptr<Plane> plane5 = std::make_shared<Plane>(Plane{ {0,0,-450}, {0,0,1} });
    plane5->setMaterial(planeSpecular1);
    std::shared_ptr<Plane> plane6 = std::make_shared<Plane>(Plane{ {0,0,450}, {0,0,-1} });
    plane6->setMaterial(planeSpecular2);

    std::shared_ptr<MultiMesh> multiMesh1 = std::make_shared<MultiMesh>(MultiMesh{ teapotObjMesh.value(), "teapot", Vector{40,0,-200} });
    std::shared_ptr<MultiMesh> multiMesh2 = std::make_shared<MultiMesh>(MultiMesh{ teapotObjMesh.value(), "teapot", Vector{-430,-100,400} });
    std::shared_ptr<MultiMesh> multiMesh3 = std::make_shared<MultiMesh>(MultiMesh{ teapotObjMesh.value(), "teapot", Vector{-150,500,200} });

    bvhAccel->addShape(triangle1->getBoundingBoxPoints(), triangle1);
    bvhAccel->addShape(triangle2->getBoundingBoxPoints(), triangle2);
    bvhAccel->addShape(multiMesh1->getBoundingBoxPoints(), multiMesh1);
    bvhAccel->addShape(multiMesh2->getBoundingBoxPoints(), multiMesh2);
    bvhAccel->addShape(multiMesh3->getBoundingBoxPoints(), multiMesh3);

    // Specular sphere
    bvhAccel->addShape(sphere1->getBoundingBoxPoints(), sphere1);
    // Reflective sphere
    bvhAccel->addShape(sphere2->getBoundingBoxPoints(), sphere2);
    bvhAccel->addShape(sphere3->getBoundingBoxPoints(), sphere3);
    // Transparent sphere
    bvhAccel->addShape(sphere4->getBoundingBoxPoints(), sphere4);

    bvhAccel->addShape(sphere5->getBoundingBoxPoints(), sphere5);
    bvhAccel->addShape(sphere6->getBoundingBoxPoints(), sphere6);
    bvhAccel->addShape(sphere7->getBoundingBoxPoints(), sphere7);

    bvhAccel->generateBVH();    
    world->scene.push_back(plane1);
    world->scene.push_back(plane2);
    world->scene.push_back(plane3);
    world->scene.push_back(plane4);
    world->scene.push_back(plane5);
    world->scene.push_back(plane6);
    world->scene.push_back(sphere1);

    world->scene.push_back(bvhAccel);

    world->ambient = std::make_shared<Ambient>();
    world->lights.push_back(
        std::make_shared<PointLight>(PointLight{ {-1000, 300, -400} }));
    world->lights[0]->setColour({ 1, 1, 1 });
    world->lights[0]->scaleRadiance(1.0f);

    world->ambient->setColour({ 1, 1, 1 });
    world->ambient->scaleRadiance(0.5f);

    Camera camera{ Point{-1000,300,250}, Point{0,0,-100}, Vector{0,1,0}, 700.0f };
    world->camera = std::make_shared<Camera>(camera);
    // Tested on 8 threads
    unsigned int numThreads = std::thread::hardware_concurrency();

    ThreadPool threadPool{ numThreads };

    world->image = std::vector<Colour>(world->height * world->width, Colour{ 0,0,0 });

    unsigned int gridN = 40;
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

    saveToFile("raytrace.bmp", world->width, world->height, world->image);

    float totalTime = timer.elapsed() - startTime;
    std::cout << "Running time is " << totalTime << std::endl;

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