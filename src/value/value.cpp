
#include <iostream>
#include <cmath>
#include <functional>
#include <set>
#include <vector>

#include "value.hpp"

using namespace micrograd;

std::vector<std::shared_ptr<Value>> Value::_sorted;
std::set<std::shared_ptr<Value>> Value::_visited;

std::ostream &micrograd::operator<<(std::ostream &s, const Value &v)
{
    s << "Value(data=" << v.get_data() << ", op=" << v.get_op() << ", grad=" << v.get_grad() << ")";
    return s;
}

void Value::backward()
{
    *_grad = 1.0;
    _topo_sort(std::make_shared<Value>(*this));
    for (auto i = _sorted.rbegin(); i != _sorted.rend(); i++)
    {
        auto val = *i;
        val->_backward(val->_l_child, val->_r_child, val);
    }

    _visited.clear();
    _sorted.clear();
}

void Value::_topo_sort(const std::shared_ptr<Value> root)
{
    if (root == nullptr)
    {
        return;
    }

    if (_visited.find(root) == _visited.end())
    {
        _visited.insert(root);
        _topo_sort(root->_l_child);
        _topo_sort(root->_r_child);
        _sorted.push_back(root);
    }
}

void Value::_to_grapviz(int &number, int parent_number) const
{
    parent_number = number;
    std::cout << "\tx" << number << " [label=<" << *this << ">];" << std::endl;

    if (_l_child)
    {
        number++;
        std::cout << "\tx" << parent_number << " -> "
                  << "x" << number << ";" << std::endl;
        _l_child->_to_grapviz(number, parent_number);
    }

    if (_r_child)
    {
        number++;
        std::cout << "\tx" << parent_number << " -> "
                  << "x" << number << ";" << std::endl;
        _r_child->_to_grapviz(number, parent_number);
    }
}

void Value::to_grapviz()
{
    std::cout << "digraph G {" << std::endl;
    int num = 0;
    _to_grapviz(num);
    std::cout << "}" << std::endl;
}

Value::Value(const Value &other)
{
    _data = std::make_unique<float>(*other._data);

    _grad = std::make_unique<float>(*other._grad);

    _backward = other._backward;

    _l_child = other._l_child;
    _r_child = other._r_child;

    _op = other._op;
}

Value &Value::operator=(const Value &other)
{
    _data = std::make_unique<float>(*other._data);
    _grad = std::make_unique<float>(*other._grad);
    _backward = other._backward;
    _l_child = other._l_child;
    _r_child = other._r_child;
    _op = other._op;
    return *this;
}

Value Value::operator+(const Value &other)
{
    Value out = Value(*_data + *other._data, std::make_shared<Value>(*this), std::make_shared<Value>(other), "+");

    out._backward = [](const std::shared_ptr<Value> thiz, const std::shared_ptr<Value> other, const std::shared_ptr<Value> out)
    {
        *thiz->_grad += *out->_grad;
        *other->_grad += *out->_grad;
    };

    return out;
}

Value Value::operator*(const Value &other)
{
    auto out = Value(*_data * *other._data, std::make_shared<Value>(*this), std::make_shared<Value>(other), "*");
    out._backward = [](const std::shared_ptr<Value> thiz, const std::shared_ptr<Value> other, const std::shared_ptr<Value> out)
    {
        *thiz->_grad += *other->_data * *out->_grad;
        *other->_grad += *thiz->_data * *out->_grad;
    };

    return out;
}

Value Value::tanh()
{
    float x = *_data;
    float e = std::exp(2 * x);
    float t = (e - 1) / (e + 1);
    auto out = Value(t, std::make_shared<Value>(*this), nullptr, "tanh");

    out._backward = [t](const std::shared_ptr<Value> thiz, const std::shared_ptr<Value> other, const std::shared_ptr<Value> out)
    {
        *thiz->_grad += (1 - std::pow(t, 2)) * *out->_grad;
    };

    return out;
}

Value Value::relu()
{
    auto out = Value(*_data < 0 ? 0 : *_data, std::make_shared<Value>(*this), nullptr, "ReLU");

    out._backward = [](const std::shared_ptr<Value> thiz, const std::shared_ptr<Value> other, const std::shared_ptr<Value> out)
    {
        *thiz->_grad = (*out->_data > 0) * *out->_grad;
    };

    return out;
}

