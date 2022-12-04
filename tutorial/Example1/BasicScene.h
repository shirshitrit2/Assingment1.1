#pragma once

#include "Scene.h"
#include <min_heap.h>

#include <utility>
#include <decimate_callback_types.h>

class BasicScene : public cg3d::Scene
{
public:
    explicit BasicScene(std::string name, cg3d::Display* display) : Scene(std::move(name), display) {};
    void Init(float fov, int width, int height, float near, float far);
    void Update(const cg3d::Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model) override;
    void KeyCallback(cg3d::Viewport* _viewport, int x, int y, int key, int scancode, int action, int mods) override;
    void reset() ;
    void decrease() ;
    void increase();
    void simplify();
    void create_p();
    void create_q_tag();
    void create_v_tag();
    void costs_positions_calculator();
    bool our_collapse_edge();



private:
    std::shared_ptr<Movable> root;
    std::shared_ptr<cg3d::Model> cyl, sphere1 ,cube;
    Eigen::VectorXi EMAP;
    Eigen::MatrixXi F,E,EF,EI;
    Eigen::VectorXi EQ;
    // If an edge were collapsed, we'd collapse it to these points:
    Eigen::MatrixXd V, C, N,T;
    igl::min_heap< std::tuple<double,int,int> > Q;
    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    std::vector<std::vector<std::shared_ptr<cg3d::Mesh>>> PreviousMeshes;
    int meshIndex=0;
    int num_collapsed;
    std::vector<Eigen::MatrixXd > P, Q_tag;
    std::map<int, Eigen::MatrixXd> V_tag;
    igl::decimate_cost_and_placement_callback our_cost_and_placement;



};
