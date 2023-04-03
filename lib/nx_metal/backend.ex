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
  def from_binary(%T{type: {type, bsize}, shape: shape} = out, binary, _opts) do
    {:ok, ref} = NIF.from_binary(binary, type, bsize, shape)
    to_nx(ref, out)
  end

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

  @impl true
  def add(out, a, b) do
    {:ok, ref} = NIF.add(from_nx(a), from_nx(b))
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
