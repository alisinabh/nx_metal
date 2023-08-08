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
    assert_equals(Nx.eye(40), Nx.eye(40, backend: Nx.BinaryBackend))

    assert_equals(Nx.eye({30, 13, 9}), Nx.eye({30, 13, 9}, backend: Nx.BinaryBackend))

    assert_equals(
      Nx.eye({10, 9, 8, 7, 6, 5}),
      Nx.eye({10, 9, 8, 7, 6, 5}, backend: Nx.BinaryBackend)
    )
  end

  Enum.map(NxMetal.NIF.bin_ops() -- [:pow], fn op ->
    test "binary operation - #{op}" do
      fuzz do
        a = random_tensor({:u, 32}, random_shape())
        b = random_tensor({:u, 32}, a.shape)

        a_mtl = Nx.backend_copy(a, NxMetal.Backend)
        b_mtl = Nx.backend_copy(b, NxMetal.Backend)

        orig_result = apply(Nx, unquote(op), [a, b])
        metal_result = apply(Nx, unquote(op), [a_mtl, b_mtl])

        assert_equals(orig_result, metal_result)
      end
    end
  end)

  test "binary operation - pow" do
    fuzz do
      a_mtl = Nx.tensor([2, 4, 8, 9, 29])
      b_mtl = Nx.tensor([2, 3, 4, 5, 6])

      metal_result = Nx.pow(a_mtl, b_mtl)

      assert Nx.to_list(metal_result) == [4, 64, 4096, 59048, 594_822_656]
    end
  end
end
