defmodule NxMetal.NIF do
  @on_load :load_nif

  def load_nif do
    path =
      :code.priv_dir(:nx_metal)
      |> Path.join("nx_metal_nif")

    case :erlang.load_nif(to_charlist(path), 0) do
      :ok ->
        :ok

      {:error, reason} ->
        {:error, {:load_nif_failed, reason}}
    end
  end

  def metal_device_name do
    :erlang.nif_error(:nif_not_loaded)
  end

  def init_metal_device do
    :erlang.nif_error(:nif_not_loaded)
  end

  def create_tensor(_dev, _tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def tensor_to_list(_tensor) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def from_binary(_binary, _bitsize, _shape) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_binary(_ref, _limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def eye(_, _, _) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def add(_, _) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
