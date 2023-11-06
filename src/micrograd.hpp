#include <iostream>
#include <cmath>
#include <functional>
#include <set>
#include <vector>

namespace micrograd
{
    class Value
    {
        float *_data;
        float *_grad;
        const Value *_l_child;
        const Value *_r_child;

        std::string _op;
        std::function<void(const Value *, const Value *, const Value *)> _backward =
            [](const Value *l_child, const Value *r_child, const Value *out)
        { return; };

        void _to_grapviz(int &number, int parent_number = 0) const;

        static std::vector<const Value *> _sorted;
        static std::set<const Value *> _visited;
        static void _topo_sort(const Value *root);

    public:
        Value(float data, const Value *l_child, const Value *r_child, std::string op = "")
        {
            this->_data = new float(data);
            this->_grad = new float(0);
            this->_l_child = l_child;
            this->_r_child = r_child;
            this->_op = op;
        }
        Value(float data, std::string op = "") : Value(data, nullptr, nullptr, op){};
        Value() : Value(0, nullptr, nullptr, ""){};

        ~Value()
        {
            delete _data;
            delete _grad;
        }
        Value(const Value &other);

        Value(Value &&source)
        {
            std::cout << "move constructor" << std::endl;
            this->_data = source._data;
            this->_grad = source._grad;
            this->_l_child = source._l_child;
            this->_r_child = source._r_child;
            this->_op = source._op;
            this->_backward = source._backward;

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

    class IModule
    {
    public:
        virtual std::vector<Value *> get_parameters() = 0;
        virtual Value operator()(std::vector<Value> &x) = 0;
    };

    class Module : IModule
    {
    public:
        void zero_grad()
        {
            for (auto p : get_parameters())
            {
                p->set_grad(0);
            }
        }
    };

    class Neuron : Module
    {
        std::vector<Value *> _w;
        Value *_b;

    public:
        std::vector<Value *> get_parameters() override;
        Value operator()(std::vector<Value> &x) override;
        Neuron(int in);
        ~Neuron();
    };
}