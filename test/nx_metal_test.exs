defmodule NxMetalTest do
  use ExUnit.Case, async: true
  doctest NxMetal

  import NxMetal.TensorUtils

  alias Nx.Tensor, as: T

  setup do
    Nx.global_default_backend(NxMetal.Backend)
    :ok
  end

  test "can create new tensors" do
    assert %T{data: %NxMetal.Backend{}} = Nx.tensor([1, 2, 3], backend: NxMetal.Backend)
  end

  test "default backend" do
    assert %T{data: %NxMetal.Backend{}} = Nx.tensor([1, 2, 3])
  end

  test "can reshape tensors" do
    t =
      Nx.tensor([1, 2, 3])
      |> Nx.reshape({3, 1, 1})

    assert %T{shape: {3, 1, 1}} = t
  end

  test "eye tensors" do
    assert Nx.to_list(Nx.eye(40)) == Nx.to_list(Nx.eye(40, backend: Nx.BinaryBackend))

    assert Nx.to_list(Nx.eye({30, 13, 9})) ==
             Nx.to_list(Nx.eye({30, 13, 9}, backend: Nx.BinaryBackend))
  end

  Enum.map(NxMetal.NIF.bin_ops() -- [:pow], fn op ->
    test "binary operation - #{op}" do
      a = random_tensor({:u, 32}, random_shape())
      b = random_tensor({:u, 32}, a.shape)

      a_mtl = Nx.backend_copy(a, NxMetal.Backend)
      b_mtl = Nx.backend_copy(b, NxMetal.Backend)

      orig_result = apply(Nx, unquote(op), [a, b])
      metal_result = apply(Nx, unquote(op), [a_mtl, b_mtl])

      assert Nx.to_list(orig_result) == Nx.to_list(metal_result)
    end
  end)

  test "binary operation - pow" do
    a = random_tensor({:f, 32}, {2, 4, 6}, 100)
    b = random_tensor({:f, 32}, a.shape, 100)

    a_mtl = Nx.backend_copy(a, NxMetal.Backend)
    b_mtl = Nx.backend_copy(b, NxMetal.Backend)

    orig_result = Nx.pow(a, b)
    metal_result = Nx.pow(a_mtl, b_mtl)

    assert Nx.to_list(orig_result) == Nx.to_list(metal_result)
  end
end
