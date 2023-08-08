defmodule NxMetal.TensorUtils do
  @moduledoc false

  import ExUnit.Assertions

  def random_shape do
    Enum.map(1..4, fn _x -> Enum.random(1..5) end)
    |> List.to_tuple()
  end

  defp elem_count(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.product()
  end

  def random_tensor(type \\ {:u, 32}, shape \\ random_shape(), max \\ 999_999) do
    elements =
      Enum.map(1..elem_count(shape), fn _ ->
        case type do
          {:u, _} -> :rand.uniform(max)
          {:s, _} -> :rand.uniform(max) - :rand.uniform(max)
          {:f, _} -> :rand.uniform() * :rand.uniform(30)
        end
      end)

    Nx.tensor(elements, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
    |> Nx.as_type(type)
  end

  defmacro fuzz(x \\ 50, block) do
    quote do
      for _ <- 1..unquote(x) do
        unquote(block)
      end
    end
  end

  def assert_equals(a, b, float_threshold \\ 0.00001)

  def assert_equals(%{type: {:f, _} = type} = a, %{type: type} = b, float_threshold) do
    assert Nx.shape(a) == Nx.shape(b)
    zipped = Enum.zip(Nx.to_flat_list(a), Nx.to_flat_list(b))
    assert Enum.all?(zipped, fn {a, b} -> abs(a - b) < float_threshold end)
  end

  def assert_equals(%{type: type} = a, %{type: type} = b, _float_threshold) do
    assert Nx.to_list(a) == Nx.to_list(b)
  end

  def assert_equals(%{type: type_a}, %{type: type_b}, _float_threshold),
    do: assert(type_a == type_b)
end
