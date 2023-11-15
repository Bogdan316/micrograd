namespace micrograd
{
    template <class T>
    class IModule
    {
    public:
        virtual std::vector<std::shared_ptr<T>> get_parameters() = 0;
        virtual T operator()(std::vector<T> &x) = 0;
    };

    template <class T>
    class Module : IModule<T>
    {
    public:
        void zero_grad()
        {
            for (T p : this->get_parameters())
            {
                p->set_grad(0);
            }
        }
    };
}