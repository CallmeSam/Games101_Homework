#include <iostream>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f squeeze = Eigen::Matrix4f::Identity();
    squeeze << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -zNear * zFar,
        0, 0, 1, 0;

    float nearH = -tan(eye_fov / 2 / 180 * MY_PI) * zNear * 2;
    float nearW = nearH * aspect_ratio;

    Eigen::Matrix4f orth = Eigen::Matrix4f::Identity();
    orth << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, -(zNear + zFar) / 2,
        0, 0, 0, 1;

    Eigen::Matrix4f orthS = Eigen::Matrix4f::Identity();
    orthS << 2 / nearW, 0, 0, 0,
        0, 2 / nearH, 0, 0,
        0, 0, 2 / (zNear - zFar), 0,
        0, 0, 0, 1;

    projection = orthS * orth * squeeze;
    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = { 0, 0, 0 };
    if (payload.texture)
    {
        /*float width = payload.texture->width;
        float height = payload.texture->height;
        float x = payload.tex_coords.x();
        float y = payload.tex_coords.y();
        int texCoordX = x * width;
        int texCoordY = y * height;
        int signX = texCoordY - int(texCoordX) > 0.5f ? 1 : -1;
        int signY = texCoordY - int(texCoordY) > 0.5f ? 1 : -1;
        Eigen::Vector2f coordlt = { signX > 0 ? texCoordX : texCoordX + signX, signY > 0 ? texCoordY : texCoordY  + signY};
        Eigen::Vector2f coordrt = { signX < 0 ? texCoordX : texCoordX + signX, signY > 0 ? texCoordY : texCoordY + signY };
        Eigen::Vector2f coordlb = { signX > 0 ? texCoordX : texCoordX + signX, signY < 0 ? texCoordY : texCoordY + signY };
        Eigen::Vector2f coordrb = { signX < 0 ? texCoordX : texCoordX + signX, signY < 0 ? texCoordY : texCoordY + signY };
        Eigen::Vector3f colorlt = payload.texture->getColor(MIN(MAX(coordlt.x() / width, 0), 0.999f), MIN(MAX(coordlt.y() / height, 0), 0.999f));
        Eigen::Vector3f colorrt = payload.texture->getColor(MIN(MAX(coordrt.x() / width, 0), 0.999f), MIN(MAX(coordrt.y() / height, 0), 0.999f));
        Eigen::Vector3f colorlb = payload.texture->getColor(MIN(MAX(coordlb.x() / width, 0), 0.999f), MIN(MAX(coordlb.y() / height, 0), 0.999f));
        Eigen::Vector3f colorrb = payload.texture->getColor(MIN(MAX(coordrb.x() / width, 0), 0.999f), MIN(MAX(coordrb.y() / height, 0), 0.999f));
        float t1 = (texCoordX - coordlt.x() - 0.5f) / 1;
        Eigen::Vector3f color1 = (1 - t1) * colorlt + t1 * colorrt;
        Eigen::Vector3f color2 = (1 - t1) * colorlb + t1 * colorrb;
        float t2 = (texCoordY - coordlt.y() - 0.5f) / 1;
        return_color = (1 - t2) * color1 + t2 * color2;*/  //Ë«ÏßÐÔ²îÖµ


        // TODO: Get the texture value at the texture coordinates of the current fragment
        return_color = payload.texture->getColor(MIN(MAX(payload.tex_coords.x(), 0), 0.999f), MIN(MAX(payload.tex_coords.y(), 0), 0.999f));
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{ {20, 20, 20}, {500, 500, 500} };
    auto l2 = light{ {-20, 20, 0}, {500, 500, 500} };

    std::vector<light> lights = { l1, l2 };
    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
    Eigen::Vector3f eye_pos{ 0, 0, 10 };

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = { 0, 0, 0 };

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        //auto ambient = Eigen::Vector3f(ka.x() * amb_light_intensity.x() * color.x(), ka.y() * amb_light_intensity.y() * color.y(), ka.z() * amb_light_intensity.z() * color.z());
        //Eigen::Vector3f lightDir = (light.position - point).normalized();
        //Eigen::Vector3f diffuse = MAX(lightDir.dot(normal), 0) * light.intensity * color * kd;
        //Eigen::Vector3f viewDir = (eye_pos - point).normalized();
        //Eigen::Vector3f halfVec = (lightDir + viewDir).normalized();
        //Eigen::Vector3f specular = pow(MAX(halfVec.dot(normal), 0.f), p) * light.intensity * color * ks;
        //result_color = ambient + diffuse + specular;
    }
    result_color = texture_color / 255.f;
    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f texture_color;

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color / 255.f;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    
    for (auto& light : lights)
    {
        auto ambient = ka.cwiseProduct(amb_light_intensity);
        float r_2 = (light.position - point).dot(light.position - point);
        Eigen::Vector3f lightDir = (light.position - point).normalized();
        Eigen::Vector3f viewDir = (eye_pos - point).normalized();
        Eigen::Vector3f halfVec = (lightDir + viewDir).normalized();
        Eigen::Vector3f diffuse = std::max(lightDir.dot(normal), 0.f) * kd.cwiseProduct(light.intensity) / r_2;
        Eigen::Vector3f specular = pow(MAX(halfVec.dot(normal), 0.f), p) * ks.cwiseProduct(light.intensity) / r_2;
        result_color += diffuse + ambient + specular;
    }

    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{ {20, 20, 20}, {500, 500, 500} };
    auto l2 = light{ {-20, 20, 0}, {500, 500, 500} };

    std::vector<light> lights = { l1, l2 };
    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
    Eigen::Vector3f eye_pos{ 0, 0, 10 };

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    Eigen::Vector3f n = normal;
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();
    Eigen::Vector3f t = Eigen::Vector3f(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    Eigen::Vector3f b = n.cross(t);
    Eigen::Vector3f norm = { 0, 0, 1 };

    Eigen::Matrix3f TBN;
    TBN << t.x(), b.x(), n.x(),
        t.y(), b.y(), n.y(),
        t.z(), b.z(), n.z();

    float width = payload.texture->width;
    float height = payload.texture->height;
    float u = payload.tex_coords.x();
    float v = payload.tex_coords.y();
    float du = kn * kh * (payload.texture->getColor(u + 1 / width, v).norm() - payload.texture->getColor(u, v).norm());
    float dv = kn * kh * (payload.texture->getColor(u, v + 1 / height).norm() - payload.texture->getColor(u, v).norm());
    norm = { -du, -dv, 1 };
    norm = (TBN * norm).normalized();

    point += kn * normal * payload.texture->getColor(u, v).norm();
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)


    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        auto ambient = ka.cwiseProduct(amb_light_intensity);
        float r_2 = (light.position - point).dot(light.position - point);
        Eigen::Vector3f lightDir = (light.position - point).normalized();
        Eigen::Vector3f viewDir = (eye_pos - point).normalized();
        Eigen::Vector3f halfVec = (lightDir + viewDir).normalized();
        Eigen::Vector3f diffuse = std::max(lightDir.dot(norm), 0.f) * kd.cwiseProduct(light.intensity) / r_2;
        Eigen::Vector3f specular = pow(MAX(halfVec.dot(norm), 0.f), p) * ks.cwiseProduct(light.intensity) / r_2;
        result_color += diffuse + ambient + specular;
    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    Eigen::Vector3f n = normal;
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();
    Eigen::Vector3f t = Eigen::Vector3f(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    Eigen::Vector3f b = n.cross(t);
    Eigen::Vector3f norm = {0, 0, 1};
    //if (payload.texture)
    //{
        // TODO: Get the texture value at the texture coordinates of the current fragment
        //norm = payload.texture->getColor(MIN(MAX(payload.tex_coords.x(), 0), 0.9999f), MIN(MAX(payload.tex_coords.y(), 0), 0.9999f));
        //norm /= 255.f;
        //norm = norm * 2.0;
        //norm(0) = norm(0) - 1.f;
        //norm(1) = norm(1) - 1.f;
        //norm(2) = norm(2) - 1.f;
        //norm.normalize();
    //}
        Eigen::Matrix3f TBN;
        //TBN << t, b, n;
        TBN << t.x(), b.x(), n.x(),
            t.y(), b.y(), n.y(),
            t.z(), b.z(), n.z();
        //norm = (TBN * norm).normalized();

        float width = payload.texture->width;
        float height = payload.texture->height;
        float u = payload.tex_coords.x();
        float v = payload.tex_coords.y();
        float du = kn * kh * (payload.texture->getColor(u + 1 / width, v).norm() - payload.texture->getColor(u, v).norm());
        float dv = kn * kh * (payload.texture->getColor(u, v + 1 / height).norm() - payload.texture->getColor(u, v).norm());
        norm = { -du, -dv, 1 };
        norm = (TBN * norm).normalized();
    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
        Eigen::Vector3f result_color = { 0, 0, 0 };
        /*for (auto& light : lights)
        {
            auto ambient = Eigen::Vector3f(ka.x() * color.x(), ka.y() * color.y(), ka.z() * color.z());
            Eigen::Vector3f lightDir = (light.position - point).normalized();
            float lengthSquare = pow((light.position - point).size(), 2);
            Eigen::Vector3f diffuse = MAX(lightDir.dot(norm), 0) * Eigen::Vector3f(color.x() * kd.x(), color.y() * kd.y(), color.z() * kd.z());
            Eigen::Vector3f viewDir = (eye_pos - point).normalized();
            Eigen::Vector3f halfVec = (lightDir + viewDir).normalized();
            Eigen::Vector3f specular = pow(MAX(halfVec.dot(norm), 0.f), p) * Eigen::Vector3f(color.x() * ks.x(), color.y() * ks.y(), color.z() * ks.z());
            result_color += diffuse + ambient + specular;
        }*/

    result_color = norm;

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "spot_texture.png";//"hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = texture_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
