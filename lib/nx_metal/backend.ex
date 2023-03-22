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

  def from_binary(%T{type: {_, bsize}, shape: shape} = tensor, binary, _opts) do
    {:ok, ref} = NIF.from_binary(binary, bsize, shape)
    to_nx(ref, tensor)
  end

  def to_binary(%T{data: %B{ref: ref}, type: {_, bits}}, limit) do
    {:ok, bin} = NIF.to_binary(ref, div(limit * bits, 8))
    bin
  end

  def eye(%T{shape: shape, type: {type, bsize}} = out, _backend_options) do
    {:ok, ref} = NIF.eye(type, bsize, shape)

    to_nx(ref, out)
  end

  @impl true
  def add(out, %T{data: %B{ref: a_ref}}, %T{data: %B{ref: b_ref}}) do
    {:ok, ref} = NIF.add(a_ref, b_ref)
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
      ~c"#Ref<" ++ rest = :erlang.ref_to_list(ref)

      Inspect.Algebra.concat([
        "NxMetal.Backend<device:#{NIF.metal_device_name()}, #{List.to_string(rest)}",
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
