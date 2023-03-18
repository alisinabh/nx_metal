defmodule NxMetal.Backend do
  defstruct [:ref]

  alias Nx.Tensor, as: T

  alias NxMetal.NIF
  alias __MODULE__, as: B

  @behaviour Nx.Backend

  @impl true
  def init(_opts) do
    {:ok, device, name} = NIF.init_metal_device()
    %{device: %{ref: device, name: name}} |> IO.inspect()
  end

  @impl true
  def iota(tensor, axis, opts) do
    IO.inspect(tensor)
    IO.inspect(axis)
    IO.inspect(opts)
  end

  def from_binary(tensor, binary, opts) do
    {:ok, ref} = NIF.from_binary(binary)
    to_nx(ref, tensor)
  end

  def to_binary(%T{data: %B{ref: ref}, type: {_, bits}}, limit) do
    {:ok, bin} = NIF.to_binary(ref, div(limit * bits, 8))
    bin
  end

  def eye(%{shape: shape, type: {t, bsize}} = out, _backend_options) do
    shape_size = tuple_size(shape)
    x = elem(shape, shape_size - 2)
    y = elem(shape, shape_size - 1)

    IO.inspect(shape)

    total_elements = shape |> Tuple.to_list() |> Enum.product()

    {:ok, ref} = NIF.eye(total_elements, x, y, t, bsize)

    to_nx(ref, out)
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  if Application.compile_env(:torchx, :add_backend_on_inspect, true) do
    defp maybe_add_signature(result, %T{data: %B{ref: ref}}) do
      Inspect.Algebra.concat([
        "NxMetal.Backend(#{NIF.metal_device_name()}, #{inspect(ref)})",
        Inspect.Algebra.line(),
        result
      ])
    end
  else
    defp maybe_add_signature(result, _tensor) do
      result
    end
  end

  defp to_nx(ref, %T{} = t) when is_reference(ref) do
    %{t | data: %B{ref: ref}}
  end
end
