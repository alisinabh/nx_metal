defmodule HelloMetalNif do
  @on_load :load_nif

  defp load_nif do
    :erlang.load_nif('./priv/hello_metal_nif', 0)
  end

  def hello do
    raise "NIF hello/0 not loaded"
  end

  def init_metal do
    raise "NIF init_metal/0 not loaded"
  end

  def create_tensor(_dev, _tensor) do
    raise "NIF create_tensor/1 not loaded"
  end

  def tensor_to_list(_tensor) do
    raise "NIF tensor_to_list/1 not loaded"
  end
end
