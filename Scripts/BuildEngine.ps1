# MIT License

# Copyright (c) 2020 Jean Philippe

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#Requires -PSEdition Core

param (
    [Parameter(HelpMessage = "Configuration type to build")]
    [ValidateSet('Debug', 'Release')]
    [string[]] $Configurations = @('Debug', 'Release'),

    [Parameter(HelpMessage = "Whether to run build, default to True")]
    [bool] $RunBuilds = $True,

    [Parameter(HelpMessage = "Whether to run clang format to format the code, default to True")]
    [bool] $RunClangFormat = $True,

    [Parameter(HelpMessage = "Whether to check code formatting correctness, default to False")]
    [bool] $VerifyFormatting = $False,

    [Parameter(HelpMessage = "VS version use to build, default to 2022")]
    [ValidateSet(2022)]
    [int] $VsVersion = 2022,

    [Parameter(HelpMessage = "Build Launcher only")]
    [switch] $LauncherOnly
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot Shared.ps1)

$cMakeProgram = Find-CMake
if ($cMakeProgram) {
    Write-Host "CMake program found..."
}
else {
    throw 'CMake program not found'
}

if ($IsWindows) {
    $nugetProgram = Find-NuGet
    if ($nugetProgram) {
        Write-Host "NuGet program found at: $nugetProgram"
    }
    else {
        Write-Warning "NuGet program not found. Attempting to download and install NuGet..."
        Setup-NuGet

        $nugetProgram = Find-NuGet
        if ($nugetProgram) {
            Write-Host "NuGet installed successfully at: $nugetProgram"
        }
        else {
            throw 'Nuget program not found'
        }
    }

    #Add NuGet to the PATH for the current session if it's not already there
    $installPath = Split-Path -Path $nugetProgram -Parent
    if ($env:PATH -notlike "*$installPath*") {
        $env:PATH = "$installPath;$env:PATH"
    }
}


if(-Not $LauncherOnly) {
    $RepoRoot = [IO.Path]::Combine($PSScriptRoot, "..")
    Write-Host "Ensuring submodules are initialized and updated..."
    & git -C $RepoRoot submodule update --init --recursive
    
    Write-Host "Configuring Vulkan-Header submodule..."
    
    $ExternalVulkanHeadersDir = Join-Path -Path $RepoRoot -ChildPath "__externals/Vulkan-Headers"
    $ExternalVulkanHeadersOutputDir = Join-Path -Path $ExternalVulkanHeadersDir -ChildPath "build"
    $ExternalVulkanHeadersInstallDir = Join-Path -Path $ExternalVulkanHeadersOutputDir -ChildPath "install"
    
    if(-Not (Test-Path -Path $ExternalVulkanHeadersInstallDir)) {
        & $cMakeProgram -S $ExternalVulkanHeadersDir -B $ExternalVulkanHeadersOutputDir
        & $cMakeProgram --install $ExternalVulkanHeadersOutputDir --prefix $ExternalVulkanHeadersInstallDir
    }
} else {
    Write-Host "Skipping submodules initialization..."
}


function Build([string]$configuration, [int]$VsVersion , [bool]$runBuild) {

    $architecture = 'x64'

    # Check if the system supports multiple configurations
    $isMultipleConfig = $IsWindows

    # Check the system name
    if ($IsLinux) {
        $systemName = "Linux"
        $cMakeGenerator
    }
    elseif ($IsMacOS) {
        $systemName = "Darwin"
    }
    elseif ($IsWindows) {
        $systemName = "Windows"
    }
    else {
        throw 'The OS is not supported'
    }

    Write-Host "Building $systemName $architecture $configuration"

    [string]$BuildDirectoryNameExtension = If ($isMultipleConfig) { "MultiConfig" } Else { $configuration }
    [string]$BuildDirectoryName = "Result." + $systemName + "." + $architecture + "." + $BuildDirectoryNameExtension
    [string]$buildDirectoryPath = [IO.Path]::Combine($RepoRoot, $BuildDirectoryName)
    [string]$cMakeCacheVariableOverride = ""
    [string]$cMakeGenerator = ""

    # Create build directory
    if (-Not (Test-Path $buildDirectoryPath)) {
        $Null = New-Item -ItemType Directory -Path $BuildDirectoryPath -ErrorAction SilentlyContinue
    }

    # Define CMake Generator arguments
    $cMakeOptions = " -DCMAKE_SYSTEM_NAME=$systemName", " -DCMAKE_BUILD_TYPE=$configuration"
    $submoduleCMakeOptions = @{
        'ENTT'      = @("-DENTT_INCLUDE_HEADERS=ON")
        'SPDLOG'    = @("-DSPDLOG_BUILD_SHARED=OFF", "-DBUILD_STATIC_LIBS=ON", "-DSPDLOG_FMT_EXTERNAL=ON", "-DSPDLOG_FMT_EXTERNAL_HO=OFF");
        'GLFW '     = @("-DGLFW_BUILD_DOCS=OFF", "-DGLFW_BUILD_EXAMPLES=OFF", "-DGLFW_INSTALL=OFF");
        'ASSIMP'    = @("-DASSIMP_BUILD_TESTS=OFF", "-DASSIMP_INSTALL=OFF", "-DASSIMP_BUILD_SAMPLES=OFF", "-DASSIMP_BUILD_ASSIMP_TOOLS=OFF", "-DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=OFF", "-DASSIMP_BUILD_OBJ_IMPORTER=ON", "-DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT=OFF", "-DASSIMP_BUILD_OBJ_EXPORTER=ON");
        'STDUUID'   = @("-DUUID_BUILD_TESTS=OFF", "-DUUID_USING_CXX20_SPAN=ON", "-DUUID_SYSTEM_GENERATOR=OFF");
        'YAMLCPP'   = @("-DYAML_CPP_BUILD_TOOLS=OFF", "-DYAML_CPP_BUILD_TESTS=OFF", "-DYAML_CPP_FORMAT_SOURCE=OFF", "-DYAML_BUILD_SHARED_LIBS=OFF");
        'FRAMEWORK' = @("-DBUILD_FRAMEWORK=ON");
        'VULKAN_LOADER' = @("-DVULKAN_HEADERS_INSTALL_DIR=$ExternalVulkanHeadersInstallDir", "-DUSE_MASM=OFF", "-DUSE_GAS=OFF")
        'SPIRV_TOOLS' = @("-DSPIRV_SKIP_EXECUTABLES=ON", "-DSPIRV_SKIP_TESTS=ON")
        'SPIRV_CROSS' = @("-DSPIRV_CROSS_ENABLE_TESTS=OFF")
        'LAUNCHER_ONLY' = @("-DLAUNCHER_ONLY=ON")
    }

    $cMakeCacheVariableOverride = $cMakeOptions -join ' '

    # Define CMake Generator argument
    switch ($systemName) {
        "Windows" {
            switch ($VsVersion) {
                2022 {
                    $cMakeGenerator = "-G `"Visual Studio 17 2022`" -A $architecture"
                }
                Default {
                    throw 'This version of Visual Studio is not supported'
                }
            }
            $cMakeCacheVariableOverride += ' -DCMAKE_CONFIGURATION_TYPES=Debug;Release '
        }
        "Linux" {
            $cMakeGenerator = "-G `"Unix Makefiles`""

            # Set Linux build compiler
            $env:CC = '/usr/bin/gcc-11'
            $env:CXX = '/usr/bin/g++-11'
        }
        "Darwin" {
            $cMakeGenerator = "-G `"Xcode`""
            $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.FRAMEWORK -join ' '
        }
        Default {
            throw 'This system is not supported'
        }
    }

    if($LauncherOnly) {
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.LAUNCHER_ONLY -join ' '
    } else {
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.ENT -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.SPDLOG -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.ASSIMP -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.STDUUID -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.YAMLCPP -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.VULKAN_LOADER -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.SPIRV_CROSS -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.SPIRV_TOOLS -join ' '
        $cMakeCacheVariableOverride += ' ' + $submoduleCMakeOptions.GLFW -join ' '
    }

    $cMakeArguments = " -S $repositoryRootPath -B $buildDirectoryPath $cMakeGenerator $cMakeCacheVariableOverride"

    # CMake Generation process
    Write-Host $cMakeArguments
    $cMakeProcess = Start-Process $cMakeProgram -ArgumentList $cMakeArguments -NoNewWindow -Wait -PassThru
    if ($cMakeProcess.ExitCode -ne 0 ) {
        throw "cmake failed generation for '$cMakeArguments' with exit code '$cMakeProcess.ExitCode'"
    }

    # CMake Build Process
    #
    if ($runBuild) {
        if ($cMakeGenerator -like 'Visual Studio*') {
            # With a Visual Studio Generator, `msbuild.exe` is used to run the build. By default, `msbuild.exe` will
            # launch worker processes to opportunistically re-use for subsequent builds. To cause the worker processes
            # to exit at the end of the main process, pass `-nodeReuse:false`.
            $buildToolOptions = '-nodeReuse:false'
        }

        $buildArguments = "--build $buildDirectoryPath --config $configuration"
        if ($buildToolOptions) {
            $buildArguments = $buildArguments, $buildToolOptions -join " --"
        }

        $buildProcess = Start-Process $cMakeProgram -ArgumentList $buildArguments -NoNewWindow -PassThru

        # Grab the process handle. When using `-NoNewWindow`, retrieving the ExitCode can return null once the process
        # has exited. See:
        # https://stackoverflow.com/questions/44057728/start-process-system-diagnostics-process-exitcode-is-null-with-nonewwindow
        $processHandle = $buildProcess.Handle
        $buildProcess.WaitForExit();
        if ($buildProcess.ExitCode -ne 0) {
            throw "cmake failed build for '$buildArguments' with exit code '$buildProcess.ExitCode'"
        }
    }
}


if(-Not $LauncherOnly) {

    # Run Clang format
    if ($RunClangFormat) {
        [string]$clangFormatScript = Join-Path $PSScriptRoot -ChildPath "ClangFormat.ps1"
        [string[]]$srcDirectories = @(
            (Join-Path $repositoryRootPath -ChildPath "ZEngine"),
            (Join-Path $repositoryRootPath -ChildPath "Tetragrama")
            (Join-Path $repositoryRootPath -ChildPath "Resources/Shaders")
        )
    
        foreach ($directory in $srcDirectories) {
            & pwsh -File $clangFormatScript -SourceDirectory $directory -RunAsCheck:$VerifyFormatting
    
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Stopped build process..." -ErrorAction Stop
            }
        }
    }


    # Run Shader Compilation
    foreach ($config in $Configurations) {
        $shaderCompileScript = Join-Path $PSScriptRoot -ChildPath "ShaderCompile.ps1"
        & pwsh -File $shaderCompileScript -Configuration:$config -ForceRebuild:$true
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Stopped build process..." -ErrorAction Stop
    }
}

# Run Engine Build
foreach ($config in $Configurations) {
    Build $config $VsVersion $RunBuilds
}
