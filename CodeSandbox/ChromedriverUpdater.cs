using System.IO.Compression;
using System.Runtime.InteropServices;
using Newtonsoft.Json;
using QuickType;

namespace CodeSandbox;

public class ChromedriverUpdater
{
    private static string endPointPrefix = "https://storage.googleapis.com/chrome-for-testing-public";
    private OSPlatform platform;
    private Architecture architecture;
    private string home;
    private string projectDirectory;

    public ChromedriverUpdater()
    {
        platform = GetOsPlatform();
        architecture = RuntimeInformation.OSArchitecture;
        home = GetPathToHome();
        projectDirectory = Path.Combine(GetPathToHome(), "RiderProjects/KnopkaDocs");
    }

    public void Run()
    {
        var link = GetDownloadLink(GetPlatformIdentifier());
        var tmp = EnsureTemporaryDirectory();
        var fileName = "chromedriver.zip";
        Download(link, tmp, fileName);
        ExtractAndDeleteSource(Path.Combine(tmp, fileName), tmp);
        var destinationDirectoryName = GetDestinationDirectoryName();
        EnsureDestinationDirectoryName(destinationDirectoryName);
        MoveFiles(tmp, destinationDirectoryName);
        DeleteTemporaryDirectory(tmp);
    }
    
    private void MoveFiles(string tmp, string destinationDirectoryName)
    {
        var extracted = Directory.GetDirectories(tmp).Single();
        var files = Directory.GetFiles(extracted).ToArray();
        var pathToChromedriver = Path.Combine(
            projectDirectory, "server/Assemblies/chromedriver", destinationDirectoryName);
        foreach (var file in files)
        {
            var fileInfo = new FileInfo(file);
            fileInfo.MoveTo(Path.Combine(pathToChromedriver, fileInfo.Name), true);
        }
    }

    void EnsureDestinationDirectoryName(string driverPath)
    {
        if (!Directory.Exists(driverPath))
            Directory.CreateDirectory(driverPath);
    }

    void DeleteTemporaryDirectory(string path)
    {
        Directory.Delete(path, true);
    }

    string EnsureTemporaryDirectory()
    {
        var tmp = Path.Combine(home, "chromedriverTmp");
        if (Directory.Exists(tmp))
            Directory.Delete(tmp, true);
        Directory.CreateDirectory(tmp);
        return tmp;
    }

    void Download(Uri link, string pathToSave, string fileName)
    {
        using var httpClient = new HttpClient();
        using var stream = httpClient.GetStreamAsync(link);
        using var fileStream = new FileStream(Path.Combine(pathToSave, fileName), FileMode.OpenOrCreate);
        stream.Result.CopyTo(fileStream);
    }

    void ExtractAndDeleteSource(string pathToFile, string destination)
    {
        ZipFile.ExtractToDirectory(pathToFile, destination);
        var sourceFile = new FileInfo(pathToFile);
        sourceFile.Delete();
    }

    string GetDestinationDirectoryName()
    {
        return architecture switch
        {
            Architecture.Arm64 when platform == OSPlatform.OSX => "osx-arm64",
            Architecture.X64 when platform == OSPlatform.OSX => "osx",
            Architecture.X64 when platform == OSPlatform.Linux => "linux",
            _ => throw new ArgumentOutOfRangeException($"Unsupported OSPlatform [{platform}] " +
                                                       $"and arhitecture [{RuntimeInformation.OSArchitecture}]")
        };
    }

    string GetPathToHome()
    {
        if (platform == OSPlatform.Linux)
            return $"/home/{Environment.UserName}";
        if (platform == OSPlatform.OSX)
            return $"/Users/{Environment.UserName}";
        throw new Exception();
    }

    Uri GetDownloadLink(Platform platform)
    {
        var url =
            "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json";
        using var httpClient = new HttpClient();
        httpClient.GetStreamAsync(url);
        var response = httpClient.GetAsync(url);
        var result = response.Result;
        if (!result.IsSuccessStatusCode)
            throw new InvalidOperationException("Что-то пошло не так");
        var version = JsonConvert.DeserializeObject<Welcome>(result.Content.ReadAsStringAsync().Result, new PlatformConverter());
        var versionWithDownload =  version.Channels.Stable.Downloads.Chromedriver;
        var downloadLink = versionWithDownload.Single(x => x.Platform == platform).Url;
        return downloadLink;
    }
    

    Platform GetPlatformIdentifier()
    {
        return architecture switch
        {
            Architecture.Arm64 when platform == OSPlatform.OSX => Platform.MacArm64,
            Architecture.X64 when platform == OSPlatform.OSX => Platform.MacX64,
            Architecture.X64 when platform == OSPlatform.Linux => Platform.Linux64,
            Architecture.X64 when platform == OSPlatform.Windows => Platform.Win64,
            _ => throw new ArgumentOutOfRangeException($"Unsupported combination of OSPlatform [{platform}] " +
                                                       $"and arhitecture [{RuntimeInformation.OSArchitecture}]")
        };
    }

    OSPlatform GetOsPlatform()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return OSPlatform.Linux;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return OSPlatform.OSX;
        throw new ArgumentOutOfRangeException($"Unsupported platform {RuntimeInformation.OSDescription}");
    }
}

