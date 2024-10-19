namespace QuickType
{
    using System;
    using System.Globalization;
    using Newtonsoft.Json;
    using Newtonsoft.Json.Converters;

    public partial class Welcome
    {
        [JsonProperty("timestamp")]
        public DateTimeOffset Timestamp { get; set; }

        [JsonProperty("channels")]
        public Channels Channels { get; set; }
    }

    public partial class Channels
    {
        [JsonProperty("Stable")]
        public Beta Stable { get; set; }

        [JsonProperty("Beta")]
        public Beta Beta { get; set; }

        [JsonProperty("Dev")]
        public Beta Dev { get; set; }

        [JsonProperty("Canary")]
        public Beta Canary { get; set; }
    }

    public partial class Beta
    {
        [JsonProperty("channel")]
        public string Channel { get; set; }

        [JsonProperty("version")]
        public string Version { get; set; }

        [JsonProperty("revision")]
        [JsonConverter(typeof(ParseStringConverter))]
        public long Revision { get; set; }

        [JsonProperty("downloads")]
        public Downloads Downloads { get; set; }
    }

    public partial class Downloads
    {
        [JsonProperty("chrome")]
        public Chrome[] Chrome { get; set; }

        [JsonProperty("chromedriver")]
        public Chrome[] Chromedriver { get; set; }

        [JsonProperty("chrome-headless-shell")]
        public Chrome[] ChromeHeadlessShell { get; set; }
    }

    public partial class Chrome
    {
        [JsonProperty("platform")]
        public Platform Platform { get; set; }

        [JsonProperty("url")]
        public Uri Url { get; set; }
    }

    public enum Platform { Linux64, MacArm64, MacX64, Win32, Win64 };

    internal static class Converter
    {
        public static readonly JsonSerializerSettings Settings = new JsonSerializerSettings
        {
            MetadataPropertyHandling = MetadataPropertyHandling.Ignore,
            DateParseHandling = DateParseHandling.None,
            Converters =
            {
                PlatformConverter.Singleton,
                new IsoDateTimeConverter { DateTimeStyles = DateTimeStyles.AssumeUniversal }
            },
        };
    }

    internal class PlatformConverter : JsonConverter
    {
        public override bool CanConvert(Type t) => t == typeof(Platform) || t == typeof(Platform?);

        public override object ReadJson(JsonReader reader, Type t, object existingValue, JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.Null) return null;
            var value = serializer.Deserialize<string>(reader);
            switch (value)
            {
                case "linux64":
                    return Platform.Linux64;
                case "mac-arm64":
                    return Platform.MacArm64;
                case "mac-x64":
                    return Platform.MacX64;
                case "win32":
                    return Platform.Win32;
                case "win64":
                    return Platform.Win64;
            }
            throw new Exception("Cannot unmarshal type Platform");
        }

        public override void WriteJson(JsonWriter writer, object untypedValue, JsonSerializer serializer)
        {
            if (untypedValue == null)
            {
                serializer.Serialize(writer, null);
                return;
            }
            var value = (Platform)untypedValue;
            switch (value)
            {
                case Platform.Linux64:
                    serializer.Serialize(writer, "linux64");
                    return;
                case Platform.MacArm64:
                    serializer.Serialize(writer, "mac-arm64");
                    return;
                case Platform.MacX64:
                    serializer.Serialize(writer, "mac-x64");
                    return;
                case Platform.Win32:
                    serializer.Serialize(writer, "win32");
                    return;
                case Platform.Win64:
                    serializer.Serialize(writer, "win64");
                    return;
            }
            throw new Exception("Cannot marshal type Platform");
        }

        public static readonly PlatformConverter Singleton = new PlatformConverter();
    }

    internal class ParseStringConverter : JsonConverter
    {
        public override bool CanConvert(Type t) => t == typeof(long) || t == typeof(long?);

        public override object ReadJson(JsonReader reader, Type t, object existingValue, JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.Null) return null;
            var value = serializer.Deserialize<string>(reader);
            long l;
            if (Int64.TryParse(value, out l))
            {
                return l;
            }
            throw new Exception("Cannot unmarshal type long");
        }

        public override void WriteJson(JsonWriter writer, object untypedValue, JsonSerializer serializer)
        {
            if (untypedValue == null)
            {
                serializer.Serialize(writer, null);
                return;
            }
            var value = (long)untypedValue;
            serializer.Serialize(writer, value.ToString());
            return;
        }

        public static readonly ParseStringConverter Singleton = new ParseStringConverter();
    }
}