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

int main()
{
    ZaxJsonParser::set_nr_indent(4);
    tensor_example_01();
    cout << "----------------------------------" << endl;
    tensor_example_02();
    cout << "----------------------------------" << endl;
    tensor_example_03();
    cout << "----------------------------------" << endl;
    tensor_example_04();
    cout << "----------------------------------" << endl;
    tensor_example_05();
    cout << "----------------------------------" << endl;
    return 0;
}
