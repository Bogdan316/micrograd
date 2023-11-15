
#include <iostream>
#include <cmath>
#include <functional>
#include <set>
#include <vector>
#include <memory>
#include <random>

namespace micrograd
{
    class Value
    {
        std::unique_ptr<float> _data;
        std::unique_ptr<float> _grad;
        std::shared_ptr<Value> _l_child;
        std::shared_ptr<Value> _r_child;

        std::string _op;
        std::function<void(const std::shared_ptr<Value>, const std::shared_ptr<Value>, const std::shared_ptr<Value>)> _backward =
            [](const std::shared_ptr<Value> l_child, const std::shared_ptr<Value> r_child, const std::shared_ptr<Value> out)
        { return; };

        void _to_grapviz(int &number, int parent_number = 0) const;

        static std::vector<std::shared_ptr<Value>> _sorted;
        static std::set<std::shared_ptr<Value>> _visited;
        static void _topo_sort(const std::shared_ptr<Value> root);

    public:
        Value(float data, std::shared_ptr<Value> l_child, std::shared_ptr<Value> r_child, std::string op = "")
        {
            this->_data = std::make_unique<float>(data);
            this->_grad = std::make_unique<float>(0);
            this->_l_child = l_child;
            this->_r_child = r_child;
            this->_op = op;
        }
        Value(float data, std::string op = "") : Value(data, nullptr, nullptr, op){};
        Value() : Value(0, nullptr, nullptr, ""){};

        ~Value()
        {
            _data.reset();
            _grad.reset();
        }
        Value(const Value &other);

        Value(Value &&source)
        {
            this->_data = std::make_unique<float>(*source._data);
            this->_grad = std::make_unique<float>(*source._grad);
            this->_l_child = source._l_child;
            this->_r_child = source._r_child;
            this->_op = source._op;
            this->_backward = source._backward;

            source._data.release();
            source._grad.release();

            source._data = nullptr;
            source._grad = nullptr;
            source._l_child = nullptr;
            source._r_child = nullptr;
        }

        float get_data() const { return *_data; }
        float get_grad() const { return *_grad; }
        void set_grad(float grad) const { *_grad = grad; }
        std::string get_op() const { return _op; }

        void backward();
        void to_grapviz();

        Value &operator=(const Value &other);
        Value operator+(const Value &other);
        Value operator*(const Value &other);
        Value tanh();
        Value relu();
    };

    std::ostream &operator<<(std::ostream &s, const Value &v);
}