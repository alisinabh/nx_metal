# NxMetal

Numrical Elixir (Nx) backend for Apple Metal framework. This will enable Nx to operate on Apple GPUs.
This also includes Apple Silicon M1/M2 chips.

*WARNING*: This library is not ready to use and just in the POC (Proof of concept phase).

## Installation

Add `nx_metal` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:nx_metal, github: "alisinabh/nx_metal"}
  ]
end
```

## Example

```elixir
iex> a = Nx.tensor([1.0, 2.1, 3.3], backend: NxMetal.Backend)
#Nx.Tensor<
  f32[3]
  NxMetal.Backend<device:Apple M1 Max, 0.110247226.3055419394.122579>
  [1.0, 2.0999999046325684, 3.299999952316284]
>
iex> b = Nx.tensor([3.1, 4.2, 8.0], backend: NxMetal.Backend)
#Nx.Tensor<
  f32[3]
  NxMetal.Backend<device:Apple M1 Max, 0.110247226.3055419394.122587>
  [3.0999999046325684, 4.199999809265137, 8.0]
>
iex> Nx.add(a, b)
#Nx.Tensor<
  f32[3]
  NxMetal.Backend<device:Apple M1 Max, 0.110247226.3055419394.122595>
  [4.099999904632568, 6.299999713897705, 11.300000190734863]
>
