defmodule NxMetal.NIF do
  @on_load :load_nif

  def load_nif do
    path =
      :code.priv_dir(:nx_metal)
      |> Path.join("priv")
      |> Path.join("nx_metal_nif")

    case :erlang.load_nif(to_charlist(path), 0) do
      :ok ->
        :ok

      {:error, reason} ->
        {:error, {:load_nif_failed, reason}}
    end
  end

  def init_metal do
    :erlang.nif_error(:nif_not_loaded)
  end
end
