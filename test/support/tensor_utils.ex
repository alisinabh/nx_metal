defmodule NxMetal.TensorUtils do
  @moduledoc false

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
end
