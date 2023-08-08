defmodule NxMetal.Backend do
  defstruct [:ref]

  require Logger

  alias Nx.Tensor, as: T

  alias NxMetal.NIF
  alias __MODULE__, as: B

  @behaviour Nx.Backend

  @impl true
  def init(opts) do
    opts
  end

  @impl true
  def from_binary(%T{type: {type, bsize}, shape: shape} = out, binary, _opts \\ []) do
    {:ok, ref} = NIF.from_binary(binary, type, bsize, shape)
    to_nx(ref, out)
  end

  @impl true
  def reshape(out, tensor), do: from_binary(out, to_binary(tensor, 0))

  @impl true
  def to_binary(%T{type: {_, bits}} = t, limit) do
    {:ok, bin} = NIF.to_binary(from_nx(t), div(limit * bits, 8))
    bin
  end

  @impl true
  def eye(%T{shape: shape, type: {type, bsize}} = out, _backend_options) do
    {:ok, ref} = NIF.eye(type, bsize, shape)

    to_nx(ref, out)
  end

  Enum.each(NIF.bin_ops() -- [:pow], fn op ->
    @impl true
    def unquote(op)(%T{type: type} = out, %T{type: type} = a, %T{type: type} = b) do
      {:ok, ref} = NIF.unquote(op)(from_nx(a), from_nx(b))
      to_nx(ref, out)
    end

    def unquote(op)(%T{type: type} = out, a, b) do
      a = Nx.as_type(a, type)
      b = Nx.as_type(b, type)
      unquote(op)(out, a, b)
    end
  end)

  @impl true
  def pow(%T{type: {:f, _} = type} = out, %T{type: type} = a, %T{type: type} = b) do
    {:ok, ref} = NIF.pow(from_nx(a), from_nx(b))
    to_nx(ref, out)
  end

  def pow(%T{type: {_, bsize} = type} = out, a, b) do
    bsize = max(min(32, bsize), 16)
    [out, a, b] = Enum.map([out, a, b], &Nx.as_type(&1, {:f, bsize}))

    pow(out, a, b)
    # Uncomment when round functionality implemented
    # |> Nx.round()
    |> Nx.as_type(type)
  end

  @impl true
  def as_type(%T{type: type}, %T{type: type} = tensor), do: tensor

  def as_type(%T{type: {type, bsize}} = out, tensor) do
    {:ok, ref} = NIF.as_type(from_nx(tensor), type, bsize)
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

  @doc false
  def from_nx(%T{data: %B{ref: device_ref}}), do: device_ref
  def from_nx(%T{} = tensor), do: Nx.backend_transfer(tensor, B) |> from_nx()

  ## All remaining callbacks

  funs = Nx.Backend.behaviour_info(:callbacks) -- Module.definitions_in(__MODULE__, :def)

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(fun)(unquote_splicing(args)) do
      Logger.warn("#{__MODULE__} unsupported operation #{unquote(fun)}")
      apply(Nx.BinaryBackend, unquote(fun), [unquote_splicing(args)])
    end
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
