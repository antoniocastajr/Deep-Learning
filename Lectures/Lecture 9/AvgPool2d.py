# this is a code fragment
# not meant to be ran independently

    class AvgPool2d(Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.kernel_size = kernel_size
            self.im2col_mat = None

        def forward(self, Xin):
            N, Cin, Hin, Win = Xin.shape
            assert(Hin % self.kernel_size == 0)
            assert(Win % self.kernel_size == 0)

            K = self.kernel_size
            S = self.kernel_size
            Hout = (Hin - K) // S + 1
            Wout = (Win - K) // S + 1
            P = Hout * Wout  # Total number of patches per image
            patch_size = Cin * K * K

            Xin_flat = Xin.reshape(-1, Cin * Hin * Win)

            if self.im2col_mat is None:
                self.im2col_mat = im2col_matrix_sparse(Xin, K, S)

            Xin_im2col = ag.spcmatmul(Xin_flat, self.im2col_mat)
            Xin_patches_flat = Xin_im2col.reshape(N, P, patch_size)

            Xin_cw_patches_flat = Xin_patches_flat.reshape(N, P, Cin, K**2)
            Xout = ag.moveaxis(ag.sum(Xin_cw_patches_flat, axis=-1), -1, -2).reshape(N, Cin, Hout, Wout) / K**2

            return Xout
