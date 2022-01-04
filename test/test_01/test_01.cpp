#include <iostream>
#include <string>
#include <map>
#include <random>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <cstring>
using namespace std;
#include "ZaxJsonParser.h"
#include "ZaxTensor.h"

struct CClassInside
{
    int x = 11;
    int y = 7;
    virtual void zax_from_json(const char* a_json, std::string* a_err_stream = 0)
    {
        x = atoi(a_json + 1);
        y = atoi(a_json + 1);
    }
    virtual int zax_to_json(char* a_json, int a_alloc_size, int a_deep = 0) const
    {
        int res = snprintf(a_json, a_alloc_size, "[%d, %d]", x, y);
        return res;
    }
};

struct CClassInside2
{
    int x = 11;
    int y = 7;
    ZAX_JSON_SERIALIZABLE(CClassInside2, JSON_PROPERTY(x), JSON_PROPERTY(y))
};

#define CClass_JOSONProps\
    JSON_PROPERTY(m_int_3d),\
    JSON_PROPERTY(title),\
    JSON_PROPERTY(x),\
    JSON_PROPERTY(intmx),\
    JSON_PROPERTY(m_float_id_of_id),\
    JSON_PROPERTY(m_int_1d_of_2d),\
    JSON_PROPERTY(inside),\
    JSON_PROPERTY(inside2),\
    JSON_PROPERTY(m_float_1d)

struct CClass
{
    float_1d m_float_1d;
    array_of_tensor_f32 m_int_1d_of_2d;
    int_3d m_int_3d;
    int_2d intmx;
    array_of_tensor_f32 m_float_id_of_id;
    CClassInside inside;
    CClassInside2 inside2;
    int x = 6;
    string title = "some title";

    ZAX_JSON_SERIALIZABLE_WDC(CClass, CClass_JOSONProps)
    CClass()
    {
        m_int_3d.resize(3,3,3);
        m_int_3d[1][1][1] = 99;
        m_int_3d[0][1][1] = 11;
        m_int_3d[1][1][0] = 110;
        m_int_3d[2][1][0] = 210;
        intmx.resize(3,3);
        unsigned int sizes[] = {3,2,5};
        m_float_id_of_id.resize(3);
        m_float_id_of_id[0].resize(sizes[0]);
        m_float_id_of_id[1].resize(sizes[1]);
        m_float_id_of_id[2].resize(sizes[2]);
        m_int_1d_of_2d.resize(2);
        m_int_1d_of_2d[0].resize(2,4);
        m_int_1d_of_2d[1].resize(3,5);
        m_float_1d.resize(11);
    }
};

void tensor_example_01()
{
    float_2d _2d(3,2);
    cout << _2d << endl;
}

void tensor_example_02()
{
    float_2d _2d(3,2);
    _2d = R"([[81,90],
              [0,2],
              [-1,3]])";
    cout << _2d << endl;
}

void tensor_example_03()
{
    float_2d _2d(3,2);
    _2d = R"([[81,90],
              [0,2],
              [-1,3]])";
    tensor_base<float>& tmpt = _2d;
    cout << tmpt << endl;
}

void tensor_example_04()
{
    float_2d _2d(3,2);
    _2d = R"([[81,90],
              [0,2],
              [-1,3]])";
    float_2d _2d2(2, 3, _2d.data());
    cout << _2d2 << endl;
}

void tensor_example_05()
{
    float_2d _2d(3,2);
    _2d = R"([[81,90],
              [0,2],
              [-1,3]])";
    float_2d _2d2(2, 3, _2d.m_data[0]);
    cout << _2d2 << endl;
}

void tensor_example_06()
{
    float_2d _2d = R"([[81,90],
              [0,2],
              [-1,3]])";
    float_2d _2d2(2, 3, _2d.m_data[0]);
    cout << _2d2 << endl;
}

void tensor_example_07()
{
    array_of_tensor_f32 m_float_id_of_id;
    m_float_id_of_id = R"([[81,90,0],
                 [0,2],
                 [-1,0,3,0,9]])";
    cout << "m_float_id_of_id: "<< m_float_id_of_id << endl;
}

void tensor_example_08()
{
    array_of_tensor_f32 m_float_id_of_id = R"([[81,90,0],
                 [0,2],
                 [-1,0,3,0,9]])";
    m_float_id_of_id[1](1)++;
    cout << "m_float_id_of_id[1][1]: "<< m_float_id_of_id[1](1) << endl;
}

void tensor_example_09()
{
    array_of_tensor_f32 m_float_id_of_id = R"([[81,90,0],
                 [0,2],
                 [-1,0,3,0,9]])";
    m_float_id_of_id[1](1)++;
    cout << "m_float_id_of_id[1][1]: "<< m_float_id_of_id[1](1) << endl;
}

void tensor_example_10()
{
    float_1d vector1;
    vector1.resize(3);
    vector1[0] = 4;
    cout << "vector1: " << vector1 << endl;
}

void tensor_example_11()
{
    float_1d vector1;
    vector1.resize(3);
    vector1[0] = 4;
    vector1 = R"([1,2,3])";
    cout << "vector1: " << vector1 << endl;
}

void tensor_example_12()
{
    tensor_4d<int> tmp1(2, 3, 4, 5);
    cout << "tmp1[0][0][0][0]:" << tmp1[0][0][0][0] << endl;
    tmp1[0][0][0][0] = 1;
    cout << "tmp1[0][0][0][0]:" << tmp1[0][0][0][0] << endl;
}

void tensor_example_13()
{
    CClass some_obj;
    some_obj = R"({"title":"some title",
    "m_int_3d":[
        [
            [0,1,0],
            [0,0,0],
            [0,2,0]],
        [
            [0,0,0],
            [0,0,3],
            [4,0,0]],
        [
            [0,0,0],
            [0,5,0],
            [0,0,6]]],
    "x":6,
    "intmx":[
        [7995588,7995588,0],
        [0,2,0],
        [-1,0,3]],
    "m_float_id_of_id":[
        [7995588,7995588,0],
        [0,2],
        [-1,0,3,0,9]],
    "inside":[11, 7],
    "m_int_1d_of_2d":[
        [
            [1,0,0,0],
            [0,2,0,0]],
        [
            [0,0,0,0,0],
            [0,0,3,4,0],
            [0,0,0,5,0]]],
    "m_float_1d": [22,0.000000,47,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,37]})";
    cout << some_obj << endl;
}

void tensor_example_14()
{
    CClass some_obj = "{\"title\": \"some other title\", \"x\": 17, \"inside\":{56}}";
    cout << some_obj << endl;
}

void tensor_example_15()
{
    float_3d _3d(2,2,3);
    float_4d _4d(2,2,2,3);
    _3d = R"([[[1.000000,2.000000,5.000000],[3.000000,4.000000,5.000000]],[[5.000000,6.000000,5.000000],[7.000000,8.000000,5.000000]]])";
    _4d = R"([
    [
        [
            [1,2,5],
            [3,4,5]],
        [
            [5,6,5],
            [7,8,5]]],
    [
        [
            [9,10,5],
            [11,12,5]],
        [
            [13,14,5],
            [15,16,5]]]])";

    cout << _3d << endl;
    _4d[0][0][0][0] = 9;
    cout << _4d << endl;
    cout << "--------------------------------" << endl;
    float_4d _4d2(1,1,24,1, _4d.m_data[0][0][0]);
    cout << _4d2 << endl;
    cout << "--------------------------------" << endl;
    float_4d _4d3;
    _4d3 = _4d;
    cout << "_4d3 and _4d numel: " << _4d3.numel() << " / " << _4d.numel() << endl;
    cout << _4d << endl;
    cout << _4d3 << endl;
    cout << "_4d3 == _4d: " << (_4d3 == _4d) << endl;
    _4d[0][0][0][0] = 10;
    cout << "_4d3 == _4d: " << (_4d3 == _4d) << endl;
    cout << "--------------------------------" << endl;
    string tmp = _4d;
    _4d3.reshape(2,6,1,2);
    cout << "_4d3 == _4d: " << (_4d3 == _4d) << endl;
    cout << _4d3 << endl;
    cout << "--------------------------------" << endl;
}

void tensor_example_16()
{
    float_3d _3d(2,2,3);
    float_4d _4d(2,2,2,3);
    _3d = R"([[[1.000000,2.000000,5.000000],[3.000000,4.000000,5.000000]],[[5.000000,6.000000,5.000000],[7.000000,8.000000,5.000000]]])";
    _4d = R"([
    [
        [
            [1,2,5],
            [3,4,5]],
        [
            [5,6,5],
            [7,8,5]]],
    [
        [
            [9,10,5],
            [11,12,5]],
        [
            [13,14,5],
            [15,16,5]]]])";

    cout << _3d << endl;
    cout << "--------------------------------" << endl;
    _4d[0][0][0][0] = 9;
    cout << _4d << endl;
    cout << "--------------------------------" << endl;
    float_4d _4d2(1,1,24,1, _4d.m_data[0][0][0]);
    cout << _4d2 << endl;
    cout << "--------------------------------" << endl;
    float_4d _4d3 = _4d;
    cout << "_4d3 and _4d numel: " << _4d3.numel() << " / " << _4d.numel() << endl;
    cout << _4d << endl;
    cout << _4d3 << endl;
    cout << "_4d3 == _4d: " << (_4d3 == _4d) << endl;
    _4d[0][0][0][0] = 10;
    cout << "_4d3 == _4d: " << (_4d3 == _4d) << endl;
    cout << "--------------------------------" << endl;
    string tmp = _4d;
    _4d3.reshape(2,6,1,2);
    cout << "_4d3 == _4d: " << (_4d3 == _4d) << endl;
    cout << _4d3 << endl;
    cout << "--------------------------------" << endl;
}

void tensor_example_17()
{
    float_3d _3d = R"([[[1.000000,2.000000,5.000000],[3.000000,4.000000,5.000000]],[[5.000000,6.000000,5.000000],[7.000000,8.000000,5.000000]]])";
    array_of_tensor_f32 ao3d;
    ao3d.resize(2);
    ao3d[0] = _3d;
    ao3d[1] = _3d;
    ao3d[1].reshape(3,2,2);
    cout << ao3d << endl;

    array_of_tensor_f32 ao3d2 = ao3d;
    cout << "ao3d == ao3d2: " << (ao3d2 == ao3d) << endl;
    ao3d[1].reshape(3,4,1);
    cout << "ao3d == ao3d2: " << (ao3d2 == ao3d) << endl;
    ao3d[1].reshape(3,2,2);
    cout << "ao3d == ao3d2: " << (ao3d2 == ao3d) << endl;
    ao3d[1](0,0,0) = 111;
    cout << "ao3d == ao3d2: " << (ao3d2 == ao3d) << endl;

    ao3d2.resize(3);
    cout << ao3d2[0].shape()[2] << endl;
    cout << "ao3d2[0]: " << ao3d2[0] << endl;
    cout << "--------------------------------" << endl;
    string tst = ao3d2;
    cout << tst << endl;
}

void tensor_example_18()
{
    array_of_tensor_f32 float_1d_of_1d_;
    float_1d_of_1d_.resize(3);
    float_1d_of_1d_[0] = R"([2,3,4])";
    float_1d_of_1d_[2] = R"([7,8,9])";
    cout << float_1d_of_1d_ << endl;
}

void tensor_example_19()
{
    tensor_f32 tensor_f32_({5,5,2});
    cout << tensor_f32_ << endl;
    cout << (tensor_f32_.m_1d?1:0) << " / " << (tensor_f32_.m_2d?1:0) << " / " << (tensor_f32_.m_3d?1:0) << " / " << (tensor_f32_.m_4d?1:0) << endl;
    tensor_f32_ = R"([2,3,4])";
    cout << "--------------------------------" << endl;
    cout << tensor_f32_ << endl;
    cout << (tensor_f32_.m_1d?1:0) << " / " << (tensor_f32_.m_2d?1:0) << " / " << (tensor_f32_.m_3d?1:0) << " / " << (tensor_f32_.m_4d?1:0) << endl;
    tensor_f32_ = R"([
    [
        [
            [1,2,5],
            [3,4,5]],
        [
            [5,6,5],
            [7,8,5]]],
    [
        [
            [9,10,5],
            [11,12,5]],
        [
            [13,14,5],
            [15,16,5]]]])";
    cout << "--------------------------------" << endl;
    cout << tensor_f32_ << endl;
    cout << (tensor_f32_.m_1d?1:0) << " / " << (tensor_f32_.m_2d?1:0) << " / " << (tensor_f32_.m_3d?1:0) << " / " << (tensor_f32_.m_4d?1:0) << endl;
    cout << tensor_f32_.shape_s() << endl;

    cout << "--------------------------------" << endl;
    tensor_f32_ = R"([
        [7995588,7995588],
        [0,2],
        [-1,0],
        [8,6]])";
    cout << tensor_f32_ << endl;
    cout << tensor_f32_.shape_s() << endl;
    tensor_f32_.resize({1, 8});
    cout << tensor_f32_ << endl;
    cout << tensor_f32_.shape_s() << endl;
    cout << "--------------------------------" << endl;
    tensor_f32_.resize({8});
    cout << tensor_f32_ << endl;
    cout << tensor_f32_.shape_s() << endl;

    tensor_f32_.resize({2,1,4});
    cout << tensor_f32_ << endl;
    cout << tensor_f32_.shape_s() << endl;
    cout << "element(1,0,2): " << tensor_f32_(1,0,2) << endl;

    tensor_f32_.squeeze(1);
    cout << tensor_f32_ << endl;
    cout << tensor_f32_.shape_s() << endl;

    tensor_f32_.unsqueeze(2).unsqueeze(1);
    cout << tensor_f32_ << endl;
    cout << tensor_f32_.shape_s() << endl;
    tensor_f32_.squeeze(1);
    cout << "--------------------------------" << endl;
    tensor_f32 tensor_f32_v = R"([[81,90,0],
                                  [0,2],
                                  [-1,0,3,0,9]])";
    *tensor_f32_.m_3d[0][0][0] = 1;
    tensor_f32_v.view({1, 8}, tensor_f32_);
    cout << tensor_f32_v << endl;
    cout << tensor_f32_v.shape_s() << endl;
    cout << "element(0,3): " << tensor_f32_v(0,3) << endl;
    cout << "--------------------------------" << endl;
    tensor_f32 tensor_f32_c = tensor_f32_;
    cout << tensor_f32_c << endl;
    cout << tensor_f32_c.shape_s() << endl;
}

void tensor_example_20()
{
    for (int i = 0 ; i < 100000; ++i)
    {
        tensor_f32* tensor_f32_2 = new tensor_f32({5,5,2});
        if (!i%10000)
            cout << *tensor_f32_2 << endl;
        delete tensor_f32_2;
    }
}

void tensor_example_21()
{
    array_of_tensor_f32 m_float_1d_of_1d;
    m_float_1d_of_1d = R"([[81,90,0],
                 [[0,2]],
                 [-1,0,3,0,9]])";
    cout << "m_float_1d_of_1d: "<< m_float_1d_of_1d << endl;
}

int main()
{
    ZaxJsonParser::set_indent(4);
    cout << "----------------------------------1----------------------------------" << endl;
    tensor_example_01();
    cout << "----------------------------------2----------------------------------" << endl;
    tensor_example_02();
    cout << "----------------------------------3----------------------------------" << endl;
    tensor_example_03();
    cout << "----------------------------------4----------------------------------" << endl;
    tensor_example_04();
    cout << "----------------------------------5----------------------------------" << endl;
    tensor_example_05();
    cout << "----------------------------------6----------------------------------" << endl;
    tensor_example_06();
    cout << "----------------------------------7----------------------------------" << endl;
    tensor_example_07();
    cout << "----------------------------------8----------------------------------" << endl;
    tensor_example_08();
    cout << "----------------------------------9----------------------------------" << endl;
    tensor_example_09();
    cout << "----------------------------------10---------------------------------" << endl;
    tensor_example_10();
    cout << "----------------------------------11---------------------------------" << endl;
    tensor_example_11();
    cout << "----------------------------------12---------------------------------" << endl;
    tensor_example_12();
    cout << "----------------------------------13---------------------------------" << endl;
    tensor_example_13();
    cout << "----------------------------------14---------------------------------" << endl;
    tensor_example_14();
    cout << "----------------------------------15---------------------------------" << endl;
    tensor_example_15();
    cout << "----------------------------------16---------------------------------" << endl;
    tensor_example_16();
    cout << "----------------------------------17---------------------------------" << endl;
    tensor_example_17();
    cout << "----------------------------------18---------------------------------" << endl;
    tensor_example_18();
    cout << "----------------------------------19---------------------------------" << endl;
    tensor_example_19();
    cout << "----------------------------------20---------------------------------" << endl;
    tensor_example_20();
    cout << "----------------------------------21---------------------------------" << endl;
    tensor_example_21();
    return 0;
}

