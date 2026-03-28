{
  "targets": [
    {
      "target_name": "cuda_addon",
      "sources": ["native/addon.cc"],
      "conditions": [
        ["OS=='win'", {
          "libraries": [
            "-l<(module_root_dir)/build/Release/obj.target/kernels.obj",
            "-l<(module_root_dir)/build/Release/obj.target/model.obj",
            "-l<(module_root_dir)/build/Release/obj.target/bpe.obj",
            "-lcudart",
            "-lcublas"
          ],
          "include_dirs": [
            "<!(echo %CUDA_PATH%)/include"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": ["/std:c++20", "/O2"]
            },
            "VCLinkerTool": {
              "AdditionalLibraryDirectories": [
                "<!(echo %CUDA_PATH%)/lib/x64"
              ]
            }
          }
        }, {
          "libraries": [
            "-L/usr/local/cuda/lib64",
            "-lcudart",
            "-lcublas",
            "<!@(pwd)/build/Release/obj.target/kernels.o",
            "<!@(pwd)/build/Release/obj.target/model.o",
            "<!@(pwd)/build/Release/obj.target/bpe.o"
          ],
          "include_dirs": [
            "/usr/local/cuda/include"
          ],
          "cflags!": ["-fno-exceptions"],
          "cflags_cc!": ["-fno-exceptions"],
          "cflags_cc": ["-std=c++17", "-O3"]
        }]
      ]
    }
  ]
}
