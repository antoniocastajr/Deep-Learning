# this is a code fragment
# not meant to be ran independently
    class Conv2d(Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1):
            super().__init__()
            self.in_channels = in_channels # Cin
            self.out_channels = out_channels # Cout
            self.kernel_size = kernel_size # K
            self.stride = stride # S

            kaiming_he_init_constant = np.sqrt(2 / (in_channels * kernel_size**2))

            # these are flattened weights
            self.weight = ag.Tensor(np.random.randn(in_channels*kernel_size**2, out_channels) * kaiming_he_init_constant)

            # bias are set to zeros initially
            self.bias = ag.Tensor(np.zeros(out_channels))

            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self.bias

            self.im2col_mat = None # im2col_mat will be cached

        def forward(self, Xin):
            # Xin: (N, Cin, Hin, Win)
            N, Cin, Hin, Win = Xin.shape
            assert(Cin == self.in_channels)

            K = self.kernel_size
            S = self.stride
            Hout = (Hin - K) // S + 1
            Wout = (Win - K) // S + 1
            P = Hout * Wout  # Total number of patches per image
            patch_size = Cin * K * K  # Size of each flattened patch
            Cout = self.out_channels

            Xin_flat = Xin.reshape(-1, Cin * Hin * Win)

            # Cache the im2col matrix
            if self.im2col_mat is None:
                self.im2col_mat = im2col_matrix_sparse(Xin, K, S)

            Xin_im2col = ag.spcmatmul(Xin_flat, self.im2col_mat)
            Xin_patches_flat = Xin_im2col.reshape(N, P, patch_size)


            Xout_flat = (Xin_patches_flat @ self.weight) + self.bias
            Xout_flat = ag.moveaxis(Xout_flat, 1, 2)
            Xout = Xout_flat.reshape(N, Cout, Hout, Wout)

            # Xout: (N, Cout, Hout, Wout)
            return Xout
