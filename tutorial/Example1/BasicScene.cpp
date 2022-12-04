#include "BasicScene.h"
#include <read_triangle_mesh.h>
#include <utility>
#include <min_heap.h>
#include "ObjLoader.h"
#include "IglMeshLoader.h"
#include "igl/read_triangle_mesh.cpp"
#include "igl/edge_flaps.h"
#include <igl/circulation.h>
#include <igl/collapse_edge.h>
#include <igl/edge_flaps.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/parallel_for.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <iostream>
#include <set>
#include <per_vertex_normals.h>
// #include "AutoMorphingModel.h"


using namespace cg3d;





void BasicScene::create_p(){
    igl::per_vertex_normals(V, F, N);
    P.resize(V.rows());
    for (int i = 0; i < V.rows(); i++) {
//        Eigen::MatrixXd n=N.row(i);
//        Eigen::MatrixXd v=V.row(i);
//        Eigen::MatrixXd d= (-(n.transpose() * v));
//        double d_value=d(0,0);
//        Eigen::DiagonalMatrix<double, 3> d_diagonal(d_value*d_value, d_value*d_value, d_value*d_value);
//        Eigen::MatrixXd q=( (v.transpose()*(n*n.transpose())*v) + (2*(d*n).transpose()*v) + d_diagonal);

        Eigen::MatrixXd n = N.row(i);
        Eigen::MatrixXd n_trans = n.transpose();
        Eigen::MatrixXd v = V.row(i);
        Eigen::MatrixXd v_trans = v.transpose();
        Eigen::MatrixXd d = -(n_trans * v);
        double d_val = d(0,0);
        Eigen::MatrixXd d_square = d * d;
        double d_val_square = d_val*d_val;

        Eigen::MatrixXd a = v_trans * (n * n_trans) * v;
        Eigen::MatrixXd b = (2 * (d_val*n).transpose())*v;

        Eigen::DiagonalMatrix<double, 3> c(d_val_square, d_val_square, d_val_square);
        Eigen::MatrixXd q = static_cast<Eigen::MatrixXd>(a + b) + static_cast<Eigen::MatrixXd>(c);

        Eigen::Matrix4d P4x4;
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                P4x4(i, j) = q(i, j);
            }
        }
        P4x4(3,0) = 0;
        P4x4(3,1) = 0;
        P4x4(3,2) = 0;
        P4x4(0, 3) = 0;
        P4x4(1, 3) = 0;
        P4x4(2, 3) = 0;
        P4x4(3,3) = 1;

        P[i]=P4x4;
    }

}

void BasicScene::create_q_tag(){
    Q_tag.resize(E.rows());
    for(int i=0;i<E.rows();i++){
        Eigen::MatrixXi e=E.row(i);
        Eigen::MatrixXd q_tag=P[e.coeff(0,0)]+P[e.coeff(0,1)];
        Q_tag[i]=q_tag;

    }

}

void BasicScene::create_v_tag(){
    for(int i = 0; i<E.rows(); i++) {

        Eigen::FullPivLU<Eigen::MatrixXd> q_matrix(Q_tag[i]);
        Eigen::Vector4d v_tag;
        if(q_matrix.isInvertible()) {
            Eigen::MatrixXd inv = q_matrix.inverse();
            Eigen::MatrixXd ones = Eigen::Vector4d(0, 0, 0, 1);
            Eigen::MatrixXd result = inv * ones;
            v_tag = result;

//            Eigen::MatrixXd vec = Eigen::Vector4d(0, 0, 0, 1);
//            v_tag = q_matrix.inverse() * vec;
        } else {
            Eigen::MatrixXd midpoint = static_cast<Eigen::MatrixXd>(0.5 * (V.row(E.row(i)[0]) + V.row(E.row(i)[1])));
            v_tag.x() = midpoint(0, 0);
            v_tag.y() = midpoint(0, 1);
            v_tag.z() = midpoint(0, 2);
            v_tag.w() = 1;
        }

        V_tag[i] = v_tag;
    }
}

void BasicScene::costs_positions_calculator(){
    create_p();
    create_q_tag();
    create_v_tag();

    our_cost_and_placement= [this] (const int e, const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXi & E, const Eigen::VectorXi &EMAP,
                                      const Eigen::MatrixXi & EF, const Eigen::MatrixXi &EI, double & cost, Eigen::RowVectorXd & p) {
        Eigen::MatrixXd q = Q_tag[e];
        Eigen::MatrixXd v = V_tag[e];
        p = Eigen::Vector3d(v(0,0), v(1,0), v(2,0));
        cost = (v.transpose() * q * v)(0,0);

    };
}

bool BasicScene::our_collapse_edge(){
    if (!igl::collapse_edge(our_cost_and_placement, V, F, E, EMAP, EF, EI, Q, EQ, C)) {
        return false;
    }
    auto curr = Q.top();
    auto cost = std::get<0>(curr);
    auto edge = std::get<1>(curr);
    std::cout<< "Edge:" << edge << ", Cost:"<<cost<<", new V position:"<<V_tag[edge]<< std::endl;
    return true;
}

void BasicScene::simplify() {
    //reset();
    costs_positions_calculator();
    if(!Q.empty()) {
        bool something_collapsed = false;
        // collapse edge
        const int max_iter = std::ceil(0.01 * Q.size());
        for (int j = 0; j < max_iter; j++) {
            if (!our_collapse_edge()) {
                break;
            }
            something_collapsed = true;
            num_collapsed++;
        }

        if (something_collapsed) {
            igl::per_vertex_normals(V, F, N);
            T = Eigen::MatrixXd::Zero(V.rows(), 2);
            std::vector<cg3d::MeshData> newMeshData;
            newMeshData.push_back({V, F, N, T});
            std::vector<std::shared_ptr<cg3d::Mesh>> newMeshList;
            for(int i=0; i<sphere1->GetMeshList().size(); i++) {
                std::shared_ptr<cg3d::Mesh> newMesh = std::make_shared<cg3d::Mesh>("new mesh", newMeshData);
                if(newMesh != nullptr) {
                    newMeshList.push_back(newMesh);
                } else {
                    newMeshList.push_back(sphere1->GetMesh(i));
                }
            }
            PreviousMeshes.push_back(newMeshList);
            meshIndex++;
            sphere1->SetMeshList(newMeshList);
        }
    }


}


void BasicScene::reset() {
    num_collapsed=0;

    auto mesh = sphere1->GetMeshList();
    V=mesh[0]->data[0].vertices;
    F = mesh[0]->data[0].faces;
    igl::edge_flaps(F, E, EMAP, EF, EI);
    C.resize(E.rows(), V.cols());
    Eigen::VectorXd costs(E.rows());
    // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
    // Q.clear();
    Q = {};
    EQ = Eigen::VectorXi::Zero(E.rows());
    {
        Eigen::VectorXd costs(E.rows());
        igl::parallel_for(E.rows(), [&](const int e) {
            double cost = e;
            Eigen::RowVectorXd p(1, 3);
            igl::shortest_edge_and_midpoint(e, V, F, E, EMAP, EF, EI, cost, p);
            C.row(e) = p;
            costs(e) = cost;
        }, 10000);
        for (int e = 0; e < E.rows(); e++) {
            Q.emplace(costs(e), e, 0);
        }
    }

}

void BasicScene::increase() {
    //reset();
    if(meshIndex!=0){
        std::vector<std::shared_ptr<cg3d::Mesh>> prevMesh= PreviousMeshes[meshIndex-1];
        sphere1->SetMeshList(prevMesh);
        meshIndex--;
    }


}

void BasicScene::decrease() {
    if(meshIndex!= PreviousMeshes.size()-1){
        std::vector<std::shared_ptr<cg3d::Mesh>> prevMesh= PreviousMeshes[meshIndex+1];
        sphere1->SetMeshList(prevMesh);
        meshIndex++;
    }
}



void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    camera = Camera::Create( "camera", fov, float(width) / height, near, far);

    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    auto daylight{std::make_shared<Material>("daylight", "shaders/cubemapShader")};
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{Model::Create("background", Mesh::Cube(), daylight)};
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();


    auto program = std::make_shared<Program>("shaders/basicShader");
    auto material{ std::make_shared<Material>("material", program)}; // empty material
//    SetNamedObject(cube, Model::Create, Mesh::Cube(), material, shared_from_this());

    material->AddTexture(0, "textures/box0.bmp", 2);
    auto sphereMesh{IglLoader::MeshFromFiles("sphere_igl", "data/sphere.obj")};
   // auto cylMesh{IglLoader::MeshFromFiles("cyl_igl","data/camel_b.obj")};
   // auto cubeMesh{IglLoader::MeshFromFiles("cube_igl","data/cube.off")};

   sphere1 = Model::Create( "sphere",sphereMesh, material);
  //  cyl = Model::Create( "cyl", cylMesh, material);
  //  cube = Model::Create( "cube", cubeMesh, material);
    sphere1->Scale(2);
   sphere1->showWireframe = true;
    sphere1->Translate({-3,0,0});
 //   cyl->Translate({3,0,0});
 //   cyl->Scale(0.12f);
 //   cyl->showWireframe = true;
//    cube->showWireframe = true;
    camera->Translate(20, Axis::Z);
    root->AddChild(sphere1);
   // root->AddChild(cyl);
  //  root->AddChild(cube);


    //igl::read_triangle_mesh("data/cube.off",V,F);
    //igl::edge_flaps(F,E,EMAP,EF,EI);
    std::cout<< "vertices: \n" << V <<std::endl;
    std::cout<< "faces: \n" << F <<std::endl;

    std::cout<< "edges: \n" << E.transpose() <<std::endl;
    std::cout<< "edges to faces: \n" << EF.transpose() <<std::endl;
    std::cout<< "faces to edges: \n "<< EMAP.transpose()<<std::endl;
    std::cout<< "edges indices: \n" << EI.transpose() <<std::endl;


    reset();
    auto mesh = sphere1->GetMeshList();
    igl::per_vertex_normals(V,F,N);
    T= Eigen::MatrixXd::Zero(V.rows(),2);
    //mesh=pickedModel->GetMeshList();
    mesh[0]->data.push_back({V,F,N,T});
    sphere1->SetMeshList(mesh);
    sphere1->meshIndex = 1;
}

    void BasicScene::Update(const Program &program, const Eigen::Matrix4f &proj, const Eigen::Matrix4f &view,
                            const Eigen::Matrix4f &model) {
        Scene::Update(program, proj, view, model);
        program.SetUniform4f("lightColor", 1.0f, 1.0f, 1.0f, 0.5f);
        program.SetUniform4f("Kai", 1.0f, 1.0f, 1.0f, 1.0f);
        //cube->Rotate(0.01f, Axis::All);
    }

    void BasicScene::KeyCallback(cg3d::Viewport *_viewport, int x, int y, int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {

            if (key == GLFW_KEY_SPACE) {
                simplify();
            }
            else if(key == GLFW_KEY_UP) {
                increase();
            }
            else if(key == GLFW_KEY_DOWN){
                decrease();
            }

        }
    }






