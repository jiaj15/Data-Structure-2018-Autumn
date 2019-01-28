
#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>
#include <queue>
#include <list>


//using namespace as cv;

// 用于判断投影是否在visual hull内部
struct Projection {
    Eigen::Matrix<float, 3, 4> m_projMat;
    cv::Mat m_image;
    const uint m_threshold = 125;

    bool outOfRange(int x, int max) {
        return x < 0 || x >= max;
    }

    bool checkRange(double x, double y, double z) {
        Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
        int indX = vec3[1] / vec3[2];
        int indY = vec3[0] / vec3[2];

        if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
            return false;
        return m_image.at<uchar>((uint) (vec3[1] / vec3[2]), (uint) (vec3[0] / vec3[2])) > m_threshold;
    }
};

// 用于index和实际坐标之间的转换
struct CoordinateInfo {
    int m_resolution;
    double m_min;
    double m_max;

    double index2coor(int index) {
        return m_min + index * (m_max - m_min) / m_resolution;
    }

    CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
            : m_resolution(resolution), m_min(min), m_max(max) {
    }
};

class Model {
public:
    typedef std::vector<std::vector<bool>> Pixel;
    typedef std::vector<Pixel> Voxel;

    Model(int resX = 100, int resY = 100, int resZ = 100);

    ~Model();

    void saveModel(const char *pFileName);

    void saveModelWithNormal(const char *pFileName);

    void loadMatrix(const char *pFileName);

    void loadImage(const char *pDir, const char *pPrefix, const char *pSuffix);

    bool checkInModel(int indX, int indY, int indZ);//check in /out model or at boundary

    void getModel();

    void getSurface();

    bool checkSurface(int indX, int indY, int indZ);

    void BFS(int indX, int indY, int indZ);

    Eigen::Vector3f getNormal(int indX, int indY, int indZ);

private:
    CoordinateInfo m_corrX;
    CoordinateInfo m_corrY;
    CoordinateInfo m_corrZ;

    int m_neiborSize;

    std::vector<Projection> m_projectionList;

    Voxel m_voxel;
    Voxel m_surface;
    Voxel m_visited;//判断是否在表面上，BFS
    Voxel m_checkinFlag;//判断是否判别过voxel
    std::vector<Eigen::Vector3i>surface_set;//store the surface points
    //std::list<PointIndex>surface_set;



};

Model::Model(int resX, int resY, int resZ)
        : m_corrX(resX, -5, 5), m_corrY(resY, -10, 10), m_corrZ(resZ, 15, 30) {
    if (resX > 100)
        m_neiborSize = resX / 100;
    else
        m_neiborSize = 1;

    m_voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, true)));
    m_surface = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, false)));
    m_visited = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, false)));
    m_checkinFlag=m_visited;

}

Model::~Model() {
}

void Model::saveModel(const char *pFileName) {
    std::ofstream fout(pFileName);
    for(int i=0;i<surface_set.size();i++) {
        Eigen::Vector3i a = surface_set[i];
        double coorX = m_corrX.index2coor(a(0));
        double coorY = m_corrY.index2coor(a(1));
        double coorZ = m_corrZ.index2coor(a(2));
        fout << coorX << ' ' << coorY << ' ' << coorZ <<std::endl;
    }

}

void Model::saveModelWithNormal(const char *pFileName) {
    std::ofstream fout(pFileName);

    for(int i=0;i<surface_set.size();i++) {
        Eigen::Vector3i a = surface_set[i];
        double coorX = m_corrX.index2coor(a(0));
        double coorY = m_corrY.index2coor(a(1));
        double coorZ = m_corrZ.index2coor(a(2));
        fout << coorX << ' ' << coorY << ' ' << coorZ << ' ';
        Eigen::Vector3f nor = getNormal(a(0),a(1), a(2));
        fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
    }

}

void Model::loadMatrix(const char *pFileName) {
    std::ifstream fin(pFileName);

    int num;
    Eigen::Matrix<float, 3, 3> matInt;
    Eigen::Matrix<float, 3, 4> matExt;
    Projection projection;
    while (fin >> num) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                fin >> matInt(i, j);

        double temp;
        fin >> temp;
        fin >> temp;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                fin >> matExt(i, j);

        projection.m_projMat = matInt * matExt;
        m_projectionList.push_back(projection);
    }
}

void Model::loadImage(const char *pDir, const char *pPrefix, const char *pSuffix) {
    int fileCount = m_projectionList.size();
    std::string fileName(pDir);
    fileName += '/';
    fileName += pPrefix;
    for (int i = 0; i < fileCount; i++) {
        std::cout << fileName + std::to_string(i) + pSuffix << std::endl;
        m_projectionList[i].m_image = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);
    }
}

bool Model::checkInModel(int indexX, int indexY, int indexZ) {
    if(!m_checkinFlag[indexX][indexY][indexZ])
    {
    double coorX = m_corrX.index2coor(indexX);
    double coorY = m_corrY.index2coor(indexY);
    double coorZ = m_corrZ.index2coor(indexZ);
    for (int i = 0; i < m_projectionList.size(); i++) {
        m_voxel[indexX][indexY][indexZ] =
                (m_voxel[indexX][indexY][indexZ]) && m_projectionList[i].checkRange(coorX, coorY, coorZ);
    }
    m_checkinFlag[indexX][indexY][indexZ]=true;
    }

    //m_visited[indexX][indexY][indexZ] = true;
    return m_voxel[indexX][indexY][indexZ];
}

//void Model::getModel() {
//    int prejectionCount = m_projectionList.size();
//
//    for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
//        for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
//            for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
//                for (int i = 0; i < prejectionCount; i++) {
//                    double coorX = m_corrX.index2coor(indexX);
//                    double coorY = m_corrY.index2coor(indexY);
//                    double coorZ = m_corrZ.index2coor(indexZ);
//                    m_voxel[indexX][indexY][indexZ] =
//                            m_voxel[indexX][indexY][indexZ] && m_projectionList[i].checkRange(coorX, coorY, coorZ);
//                }
//}

bool Model::checkSurface(int indX, int indY, int indZ) {
    int dx[6] = {-1, 0, 0, 0, 0, 1};
    int dy[6] = {0, 1, -1, 0, 0, 0};
    int dz[6] = {0, 0, 0, 1, -1, 0};

    // lambda表达式，用于判断某个点是否在Voxel的范围内
    auto outOfRange = [&](int indexX, int indexY, int indexZ) {
        return indexX < 0 || indexY < 0 || indexZ < 0
               || indexX >= m_corrX.m_resolution
               || indexY >= m_corrY.m_resolution
               || indexZ >= m_corrZ.m_resolution;
    };
    checkInModel(indX,indY,indZ);
    if (!m_voxel[indX][indY][indZ]) {
        m_surface[indX][indY][indZ] = false;

    } else {
        bool ans = false;
        for (int i = 0; i < 6; i++) {
            checkInModel(indX + dx[i], indY + dy[i], indZ + dz[i]);
            ans = ans || outOfRange(indX + dx[i], indY + dy[i], indZ + dz[i])
                  || !m_voxel[indX + dx[i]][indY + dy[i]][indZ + dz[i]];
        }
        m_surface[indX][indY][indZ] = ans;
        m_visited[indX][indY][indZ]=true;
    }
    return m_surface[indX][indY][indZ];
}

void Model::getSurface() {
    //clock_t t1=clock();
    // 邻域：上、下、左、右、前、后。
    int dx[6] = {-1, 0, 0, 0, 0, 1};
    int dy[6] = {0, 1, -1, 0, 0, 0};
    int dz[6] = {0, 0, 0, 1, -1, 0};

    // lambda表达式，用于判断某个点是否在Voxel的范围内
    auto outOfRange = [&](int indexX, int indexY, int indexZ) {
        return indexX < 0 || indexY < 0 || indexZ < 0
               || indexX >= m_corrX.m_resolution
               || indexY >= m_corrY.m_resolution
               || indexZ >= m_corrZ.m_resolution;
    };

    /*-----find a surface point-----*/
    int centerX = m_corrX.m_resolution / 2;
    int centerY = m_corrY.m_resolution / 2;
    int centerZ = m_corrZ.m_resolution / 2;

    bool issurface = false;
    while (!issurface && centerX < m_corrX.m_resolution) {
        while (!issurface && centerY < m_corrY.m_resolution) {
            while (!issurface && centerZ < m_corrZ.m_resolution) {
                bool isinside = checkInModel(centerX, centerY, centerZ);
                bool isNoutside = false;
                for (int i = 0; i < 6; i++) {
                    //checkInModel(centerX + dx[i],centerY + dy[i],centerZ + dz[i]);
                    for (int k = 0; k < m_projectionList.size(); k++) {
                        double coorX = m_corrX.index2coor(centerX + dx[i]);
                        double coorY = m_corrY.index2coor(centerY + dy[i]);
                        double coorZ = m_corrZ.index2coor(centerZ + dz[i]);
                        m_voxel[centerX + dx[i]][centerY + dy[i]][centerZ + dz[i]] =
                                (m_voxel[centerX + dx[i]][centerY + dy[i]][centerZ + dz[i]]) && m_projectionList[k].checkRange(coorX, coorY, coorZ);
                    }

                    isNoutside = isNoutside || outOfRange(centerX + dx[i], centerY + dy[i], centerZ + dz[i])
                                 || !m_voxel[centerX + dx[i]][centerY + dy[i]][centerZ + dz[i]];
                }
                if (isinside == isNoutside) {
                    issurface = true;
                    m_surface[centerX][centerY][centerZ] = true;

                } else {
                    m_surface[centerX][centerY][centerZ] = false;
                }

                centerZ++;

            }
            centerY++;
        }
        centerX++;


    }
    /*-----find a surface point end-----*/
    //std::cout<<(float(clock() - t1) / CLOCKS_PER_SEC)<<std::endl;
    BFS(centerX-1,centerY-1,centerZ-1);

}

void Model::BFS(int indX, int indY, int indZ){
    //std::ofstream fout("WithoutNormal.xyz");

    int n=1;
    int dx[3] = {-1, 0, 1};
    int dy[3] = {-1, 0, 1};
    int dz[3] = {-1, 0, 1};
    std::queue<Eigen::Vector3i>q;
    m_visited[indX][indY][indZ]=true;

    Eigen::Vector3i temp(indX,indY,indZ);
    //temp.indexX=indX;temp.indexY=indY;temp.indexZ=indZ;
    q.push(temp);
    while(!q.empty())
    {
        Eigen::Vector3i v=q.front();
        q.pop();
        surface_set.push_back(v);
//        double coorX = m_corrX.index2coor(v.indexX);
//        double coorY = m_corrY.index2coor(v.indexY);
//        double coorZ = m_corrZ.index2coor(v.indexZ);
        //fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
       // std::cout<<" "<<v(0)<<" "<<v(1)<<" "<<v(2)<<std::endl;
        for(int k=0;k<3;k++){
            for(int j=0;j<3;j++){
                for(int i =0;i<3;i++)
                {

                    Eigen::Vector3i u(v(0)+dx[i],v(1)+dy[j],v(2)+dz[k]);
                    //u.indexX=v.indexX+dx[i];u.indexY=v.indexY+dy[j];u.indexZ=v.indexZ+dz[k];
                    if(! m_visited[u(0)][u(1)][u(2)])
                    {
                        m_visited[u(0)][u(1)][u(2)]=true;
                        if(checkSurface(u(0),u(1),u(2)))
                            q.push(u);

                    }
                }
            }

        }


    }}


Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ) {
    auto outOfRange = [&](int indexX, int indexY, int indexZ) {
        return indexX < 0 || indexY < 0 || indexZ < 0
               || indexX >= m_corrX.m_resolution
               || indexY >= m_corrY.m_resolution
               || indexZ >= m_corrZ.m_resolution;
    };

    std::vector<Eigen::Vector3f> neiborList;
    std::vector<Eigen::Vector3f> innerList;

    for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
        for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
            for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++) {
                if (!dX && !dY && !dZ)
                    continue;
                int neiborX = indX + dX;
                int neiborY = indY + dY;
                int neiborZ = indZ + dZ;
                if (!outOfRange(neiborX, neiborY, neiborZ)) {
                    float coorX = m_corrX.index2coor(neiborX);
                    float coorY = m_corrY.index2coor(neiborY);
                    float coorZ = m_corrZ.index2coor(neiborZ);
                    //;
                    if(!m_visited[neiborX][neiborY][neiborZ])
                    {checkInModel(neiborX,neiborY,neiborZ);
                    m_visited[neiborX][neiborY][neiborZ]=true;
                    }
                    else if (m_surface[neiborX][neiborY][neiborZ])
                    { neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));}
                    else if (m_voxel[neiborX][neiborY][neiborZ])
                        innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
                }
            }

    Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));

    Eigen::MatrixXf matA(3, neiborList.size());
    for (int i = 0; i < neiborList.size(); i++)
        matA.col(i) = neiborList[i] - point;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
    Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
    int indexEigen = 0;
    if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
        indexEigen = 1;
    if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
        indexEigen = 2;
    Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

    Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
    for (auto const &vec : innerList)
        innerCenter += vec;
    innerCenter /= innerList.size();

    if (normalVector.dot(point - innerCenter) < 0)
        normalVector *= -1;
    return normalVector;
}



int main(int argc, char **argv) {
    clock_t t = clock();

    // 分别设置xyz方向的Voxel分辨率
    Model model(300, 300, 300);

    // 读取相机的内外参数
    model.loadMatrix("calibParamsI.txt");

    // 读取投影图片
    model.loadImage("wd_segmented", "WD2_", "_00020_segmented.png");

    // 得到Voxel模型
    //clock_t tl = clock();
    //std::cout << " load files time: " << (float(tl - t) / CLOCKS_PER_SEC) << "seconds\n";

    // model.getModel();

    std::cout << "get model done\n";
    clock_t tm = clock();
    //std::cout << " get model done time: " << (float(tm - tl) / CLOCKS_PER_SEC) << "seconds\n";

    // 获得Voxel模型的表面
    model.getSurface();
    //clock_t ts = clock();
    std::cout << "get surface done\n";
   // std::cout << " get surface done time: " << (float(ts - tm) / CLOCKS_PER_SEC) << "seconds\n";

    // 将模型导出为xyz格式
   model.saveModel("WithoutNormal.xyz");
    //clock_t tnn = clock();
    std::cout << "save without normal done\n";
    //std::cout << " save without normal done time: " << (float(tnn - ts) / CLOCKS_PER_SEC) << "seconds\n";


    model.saveModelWithNormal("WithNormal.xyz");
    //model.saveply("mesh.ply");
    clock_t tn = clock();
    std::cout << "save with normal done\n";
    std::cout << " save with normal done time: " << (float(tn - t) / CLOCKS_PER_SEC) << "seconds\n";

    system("PoissonRecon.x64 --in WithNormal.xyz --out mesh.ply");
    clock_t tt = clock();
    std::cout << "save mesh.ply done\n";
    std::cout << " PoissonRecon: " << (float(tt - tn) / CLOCKS_PER_SEC) << "seconds\n";

    t = clock() - t;
    std::cout << "total time: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

    return (0);
}


//#pragma warning(disable:4819)
//#pragma warning(disable:4244)
//#pragma warning(disable:4267)
//
//#include <time.h>
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <algorithm>
//#include <opencv2/opencv.hpp>
//#include <Eigen/Eigen>
//#include <limits>
//
////using namespace as cv;
//
//// 用于判断投影是否在visual hull内部
//struct Projection
//{
//    Eigen::Matrix<float, 3, 4> m_projMat;
//    cv::Mat m_image;
//    const uint m_threshold = 125;
//
//    bool outOfRange(int x, int max)
//    {
//        return x < 0 || x >= max;
//    }
//
//    bool checkRange(double x, double y, double z)
//    {
//        Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
//        int indX = vec3[1] / vec3[2];
//        int indY = vec3[0] / vec3[2];
//
//        if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
//            return false;
//        return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold;
//    }
//};
//
//// 用于index和实际坐标之间的转换
//struct CoordinateInfo
//{
//    int m_resolution;
//    double m_min;
//    double m_max;
//
//    double index2coor(int index)
//    {
//        return m_min + index * (m_max - m_min) / m_resolution;
//    }
//
//    CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
//            : m_resolution(resolution)
//            , m_min(min)
//            , m_max(max)
//    {
//    }
//};
//
//class Model
//{
//public:
//    typedef std::vector<std::vector<bool>> Pixel;
//    typedef std::vector<Pixel> Voxel;
//
//    Model(int resX = 100, int resY = 100, int resZ = 100);
//    ~Model();
//
//    void saveModel(const char* pFileName);
//    void saveModelWithNormal(const char* pFileName);
//    void loadMatrix(const char* pFileName);
//    void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
//    void getModel();
//    void getSurface();
//    Eigen::Vector3f getNormal(int indX, int indY, int indZ);
//
//private:
//    CoordinateInfo m_corrX;
//    CoordinateInfo m_corrY;
//    CoordinateInfo m_corrZ;
//
//    int m_neiborSize;
//
//    std::vector<Projection> m_projectionList;
//
//    Voxel m_voxel;
//    Voxel m_surface;
//};
//
//Model::Model(int resX, int resY, int resZ)
//        : m_corrX(resX, -5, 5)
//        , m_corrY(resY, -10, 10)
//        , m_corrZ(resZ, 15, 30)
//{
//    if (resX > 100)
//        m_neiborSize = resX / 100;
//    else
//        m_neiborSize = 1;
//    m_voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, true)));
//    m_surface = m_voxel;
//}
//
//Model::~Model()
//{
//}
//
//void Model::saveModel(const char* pFileName)
//{
//    std::ofstream fout(pFileName);
//
//    for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
//        for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
//            for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
//                if (m_surface[indexX][indexY][indexZ])
//                {
//                    double coorX = m_corrX.index2coor(indexX);
//                    double coorY = m_corrY.index2coor(indexY);
//                    double coorZ = m_corrZ.index2coor(indexZ);
//                    fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
//                }
//}
//
//void Model::saveModelWithNormal(const char* pFileName)
//{
//    std::ofstream fout(pFileName);
//
//    double midX = m_corrX.index2coor(m_corrX.m_resolution / 2);
//    double midY = m_corrY.index2coor(m_corrY.m_resolution / 2);
//    double midZ = m_corrZ.index2coor(m_corrZ.m_resolution / 2);
//
//    for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
//        for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
//            for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
//                if (m_surface[indexX][indexY][indexZ])
//                {
//                    double coorX = m_corrX.index2coor(indexX);
//                    double coorY = m_corrY.index2coor(indexY);
//                    double coorZ = m_corrZ.index2coor(indexZ);
//                    fout << coorX << ' ' << coorY << ' ' << coorZ << ' ';
//
//                    Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ);
//                    fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
//                }
//}
//
//void Model::loadMatrix(const char* pFileName)
//{
//    std::ifstream fin(pFileName);
//
//    int num;
//    Eigen::Matrix<float, 3, 3> matInt;
//    Eigen::Matrix<float, 3, 4> matExt;
//    Projection projection;
//    while (fin >> num)
//    {
//        for (int i = 0; i < 3; i++)
//            for (int j = 0; j < 3; j++)
//                fin >> matInt(i, j);
//
//        double temp;
//        fin >> temp;
//        fin >> temp;
//        for (int i = 0; i < 3; i++)
//            for (int j = 0; j < 4; j++)
//                fin >> matExt(i, j);
//
//        projection.m_projMat = matInt * matExt;
//        m_projectionList.push_back(projection);
//    }
//}
//
//void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
//{
//    int fileCount = m_projectionList.size();
//    std::string fileName(pDir);
//    fileName += '/';
//    fileName += pPrefix;
//    for (int i = 0; i < fileCount; i++)
//    {
//        std::cout << fileName + std::to_string(i) + pSuffix << std::endl;
//        m_projectionList[i].m_image = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);
//    }
//}
//
//void Model::getModel()
//{
//    int prejectionCount = m_projectionList.size();
//
//    for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
//        for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
//            for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
//                for (int i = 0; i < prejectionCount; i++)
//                {
//                    double coorX = m_corrX.index2coor(indexX);
//                    double coorY = m_corrY.index2coor(indexY);
//                    double coorZ = m_corrZ.index2coor(indexZ);
//                    m_voxel[indexX][indexY][indexZ] = m_voxel[indexX][indexY][indexZ] && m_projectionList[i].checkRange(coorX, coorY, coorZ);
//                }
//}
//
//void Model::getSurface()
//{
//    // 邻域：上、下、左、右、前、后。
//    int dx[6] = { -1, 0, 0, 0, 0, 1 };
//    int dy[6] = { 0, 1, -1, 0, 0, 0 };
//    int dz[6] = { 0, 0, 0, 1, -1, 0 };
//
//    // lambda表达式，用于判断某个点是否在Voxel的范围内
//    auto outOfRange = [&](int indexX, int indexY, int indexZ){
//        return indexX < 0 || indexY < 0 || indexZ < 0
//               || indexX >= m_corrX.m_resolution
//               || indexY >= m_corrY.m_resolution
//               || indexZ >= m_corrZ.m_resolution;
//    };
//
//    for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
//        for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
//            for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
//            {
//                if (!m_voxel[indexX][indexY][indexZ])
//                {
//                    m_surface[indexX][indexY][indexZ] = false;
//                    continue;
//                }
//                bool ans = false;
//                for (int i = 0; i < 6; i++)
//                {
//                    ans = ans || outOfRange(indexX + dx[i], indexY + dy[i], indexZ + dz[i])
//                          || !m_voxel[indexX + dx[i]][indexY + dy[i]][indexZ + dz[i]];
//                }
//                m_surface[indexX][indexY][indexZ] = ans;
//            }
//}
//
//Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)
//{
//    auto outOfRange = [&](int indexX, int indexY, int indexZ){
//        return indexX < 0 || indexY < 0 || indexZ < 0
//               || indexX >= m_corrX.m_resolution
//               || indexY >= m_corrY.m_resolution
//               || indexZ >= m_corrZ.m_resolution;
//    };
//
//    std::vector<Eigen::Vector3f> neiborList;
//    std::vector<Eigen::Vector3f> innerList;
//
//    for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
//        for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
//            for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
//            {
//                if (!dX && !dY && !dZ)
//                    continue;
//                int neiborX = indX + dX;
//                int neiborY = indY + dY;
//                int neiborZ = indZ + dZ;
//                if (!outOfRange(neiborX, neiborY, neiborZ))
//                {
//                    float coorX = m_corrX.index2coor(neiborX);
//                    float coorY = m_corrY.index2coor(neiborY);
//                    float coorZ = m_corrZ.index2coor(neiborZ);
//                    if (m_surface[neiborX][neiborY][neiborZ])
//                        neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
//                    else if (m_voxel[neiborX][neiborY][neiborZ])
//                        innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
//                }
//            }
//
//    Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));
//
//    Eigen::MatrixXf matA(3, neiborList.size());
//    for (int i = 0; i < neiborList.size(); i++)
//        matA.col(i) = neiborList[i] - point;
//    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
//    Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
//    int indexEigen = 0;
//    if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
//        indexEigen = 1;
//    if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
//        indexEigen = 2;
//    Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);
//
//    Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
//    for (auto const& vec : innerList)
//        innerCenter += vec;
//    innerCenter /= innerList.size();
//
//    if (normalVector.dot(point - innerCenter) < 0)
//        normalVector *= -1;
//    return normalVector;
//}
//
//
//int main(int argc, char **argv) {
//    clock_t t = clock();
//
//    // 分别设置xyz方向的Voxel分辨率
//    Model model(300, 300, 300);
//
//    // 读取相机的内外参数
//    model.loadMatrix("calibParamsI.txt");
//
//    // 读取投影图片
//    model.loadImage("E:\\jiaj1\\Documents\\Clion\\visualHull\\wd_segmented", "WD2_", "_00020_segmented.png");
//
//    // 得到Voxel模型
//    clock_t tl = clock();
//    std::cout << " load files time: " << (float(tl - t) / CLOCKS_PER_SEC) << "seconds\n";
//
//    model.getModel();
//
//    std::cout << "get model done\n";
//    clock_t tm = clock();
//    std::cout << " get model done time: " << (float(tm - tl) / CLOCKS_PER_SEC) << "seconds\n";
//
//    // 获得Voxel模型的表面
//    model.getSurface();
//    clock_t ts = clock();
//    std::cout << "get surface done\n";
//    std::cout << " get surface done time: " << (float(ts - tm) / CLOCKS_PER_SEC) << "seconds\n";
//
//    // 将模型导出为xyz格式
//    model.saveModel("WithoutNormal.xyz");
//    clock_t tnn = clock();
//    std::cout << "save without normal done\n";
//    std::cout << " save without normal done time: " << (float(tnn - ts) / CLOCKS_PER_SEC) << "seconds\n";
//
//
//    model.saveModelWithNormal("WithNormal.xyz");
//    clock_t tn = clock();
//    std::cout << "save with normal done\n";
//    std::cout << " save with normal done time: " << (float(tn - tnn) / CLOCKS_PER_SEC) << "seconds\n";
//
//    system("PoissonRecon.x64 --in WithNormal.xyz --out mesh.ply");
//    clock_t tt = clock();
//    std::cout << "save mesh.ply done\n";
//    std::cout << " save mesh.ply done time: " << (float(tt - tn) / CLOCKS_PER_SEC) << "seconds\n";
//
//    t = clock() - t;
//    std::cout << "total time: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";
//
//    return (0);
//}
