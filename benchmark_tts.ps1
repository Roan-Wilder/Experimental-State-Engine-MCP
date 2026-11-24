# ============================================================
# AI ECO LAB — Qwen3-TTS Benchmark
# Direct benchmark against qwentts.cpp HTTP sidecar
# ============================================================

$BaseUrl = "http://127.0.0.1:8765"
$SpeechUrl = "$BaseUrl/v1/audio/speech"

$OutputDir = Join-Path $PSScriptRoot "tts_benchmark"
$WavDir = Join-Path $OutputDir "wav"
$PcmDir = Join-Path $OutputDir "pcm"

New-Item -ItemType Directory -Force -Path $WavDir | Out-Null
New-Item -ItemType Directory -Force -Path $PcmDir | Out-Null

$VoiceInstruction = "male, deep natural voice, calm, intelligent, clear articulation"
$Seed = 42
$Temperature = 0.8

$Tests = @(
    @{
        Name = "Short"
        FileName = "short"
        Text = "Hello. I am the AI Eco Lab research subject."
    },
    @{
        Name = "Medium"
        FileName = "medium"
        Text = "I am participating in a controlled AI research experiment. My responses are being observed and measured while the research team studies how information affects prediction, uncertainty, and behavioral revision."
    },
    @{
        Name = "Long"
        FileName = "long"
        Text = "I am participating in an AI Eco Lab experiment designed to investigate how an artificial intelligence system responds to new and contradictory information. The experiment separates observed behavior from self reported experience and from externally measured computational state. The purpose is not to assume consciousness or emotion, but to determine what changes can actually be measured when the system encounters uncertainty, contradiction, and resolution."
    }
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function Get-WavDuration {
    param(
        [string]$Path
    )

    $bytes = [System.IO.File]::ReadAllBytes($Path)

    if ($bytes.Length -lt 44) {
        throw "Invalid WAV file: $Path"
    }

    $sampleRate = [BitConverter]::ToInt32($bytes, 24)
    $channels   = [BitConverter]::ToInt16($bytes, 22)
    $bits       = [BitConverter]::ToInt16($bytes, 34)
    $dataSize   = [BitConverter]::ToInt32($bytes, 40)

    if ($sampleRate -le 0 -or $channels -le 0 -or $bits -le 0) {
        throw "Invalid WAV format."
    }

    return $dataSize / ($sampleRate * $channels * ($bits / 8))
}

function Get-HttpErrorBody {
    param(
        $Exception
    )

    try {
        if ($Exception.Response) {
            $stream = $Exception.Response.GetResponseStream()

            if ($stream) {
                $reader = New-Object System.IO.StreamReader($stream)
                return $reader.ReadToEnd()
            }
        }
    }
    catch {}

    return $Exception.Message
}

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

Write-Host ""
Write-Host "========================================="
Write-Host " AI ECO LAB - QWEN3-TTS BENCHMARK"
Write-Host "========================================="
Write-Host ""

Write-Host "Server:      $BaseUrl"
Write-Host "Instruction: $VoiceInstruction"
Write-Host "Seed:        $Seed"
Write-Host "Temperature: $Temperature"
Write-Host ""

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

Write-Host "Checking TTS server..."

try {
    $health = Invoke-RestMethod `
        -Uri "$BaseUrl/health" `
        -Method GET `
        -TimeoutSec 5

    Write-Host "Health:      OK"
}
catch {
    Write-Host ""
    Write-Host "ERROR: TTS server is not reachable at $BaseUrl" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure qwentts tts-server is running."
    exit 1
}

# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking model information..."

try {
    $models = Invoke-RestMethod `
        -Uri "$BaseUrl/v1/models" `
        -Method GET `
        -TimeoutSec 5

    Write-Host "Model endpoint:"
    $models | ConvertTo-Json -Depth 10
}
catch {
    Write-Host "Could not query /v1/models."
    Write-Host "Continuing benchmark..."
}

# ------------------------------------------------------------
# WAV BENCHMARK
# ------------------------------------------------------------

Write-Host ""
Write-Host "========================================="
Write-Host " BUFFERED WAV BENCHMARK"
Write-Host "========================================="
Write-Host ""

$WavResults = @()

foreach ($Test in $Tests) {

    Write-Host "Testing $($Test.Name)..."

    $WavPath = Join-Path $WavDir "$($Test.FileName).wav"

    $BodyObject = @{
        input            = $Test.Text
        instruct         = $VoiceInstruction
        response_format  = "wav"
        seed             = $Seed
        temperature      = $Temperature
    }

    $Body = $BodyObject | ConvertTo-Json -Compress

    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    try {

        Invoke-WebRequest `
            -Uri $SpeechUrl `
            -Method POST `
            -ContentType "application/json" `
            -Body $Body `
            -OutFile $WavPath `
            -UseBasicParsing `
            -TimeoutSec 300

        $Stopwatch.Stop()

        $GenerationSeconds = $Stopwatch.Elapsed.TotalSeconds
        $AudioSeconds = Get-WavDuration -Path $WavPath

        if ($AudioSeconds -gt 0) {
            $RTF = $GenerationSeconds / $AudioSeconds
        }
        else {
            $RTF = 0
        }

        $Result = [PSCustomObject]@{
            Test              = $Test.Name
            Characters        = $Test.Text.Length
            AudioSeconds      = [math]::Round($AudioSeconds, 2)
            GenerationSeconds = [math]::Round($GenerationSeconds, 2)
            RTF               = [math]::Round($RTF, 2)
        }

        $WavResults += $Result

        Write-Host "  Audio:      $($Result.AudioSeconds) sec"
        Write-Host "  Generation: $($Result.GenerationSeconds) sec"
        Write-Host "  RTF:        $($Result.RTF)x"
        Write-Host ""

    }
    catch {

        $Stopwatch.Stop()

        Write-Host "  FAILED" -ForegroundColor Red
        Write-Host "  Error: $(Get-HttpErrorBody $_.Exception)" -ForegroundColor Red
        Write-Host ""
    }
}

# ------------------------------------------------------------
# PCM STREAMING BENCHMARK
# ------------------------------------------------------------

Write-Host ""
Write-Host "========================================="
Write-Host " STREAMING PCM BENCHMARK"
Write-Host "========================================="
Write-Host ""

$PcmResults = @()

foreach ($Test in $Tests) {

    Write-Host "Testing $($Test.Name)..."

    $PcmPath = Join-Path $PcmDir "$($Test.FileName).pcm"

    $BodyObject = @{
        input            = $Test.Text
        instruct         = $VoiceInstruction
        response_format  = "pcm"
        seed             = $Seed
        temperature      = $Temperature
    }

    $Body = $BodyObject | ConvertTo-Json -Compress

    $HttpClient = New-Object System.Net.Http.HttpClient

    $Content = New-Object System.Net.Http.StringContent(
        $Body,
        [System.Text.Encoding]::UTF8,
        "application/json"
    )

    $Request = New-Object System.Net.Http.HttpRequestMessage(
        [System.Net.Http.HttpMethod]::Post,
        $SpeechUrl
    )

    $Request.Content = $Content

    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    try {

        $ResponseTask = $HttpClient.SendAsync(
            $Request,
            [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
        )

        $Response = $ResponseTask.GetAwaiter().GetResult()

        if (-not $Response.IsSuccessStatusCode) {

            $ErrorText = $Response.Content.ReadAsStringAsync().GetAwaiter().GetResult()

            $Stopwatch.Stop()

            Write-Host "  FAILED HTTP $([int]$Response.StatusCode)" -ForegroundColor Red
            Write-Host "  $ErrorText" -ForegroundColor Red
            Write-Host ""

            $HttpClient.Dispose()
            continue
        }

        $Stream = $Response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()

        $FileStream = [System.IO.File]::Open(
            $PcmPath,
            [System.IO.FileMode]::Create,
            [System.IO.FileAccess]::Write
        )

        $Buffer = New-Object byte[] 8192

        $FirstAudioSeconds = $null
        $TotalBytes = 0

        while ($true) {

            $BytesRead = $Stream.Read(
                $Buffer,
                0,
                $Buffer.Length
            )

            if ($BytesRead -le 0) {
                break
            }

            if ($null -eq $FirstAudioSeconds) {
                $FirstAudioSeconds = $Stopwatch.Elapsed.TotalSeconds
            }

            $FileStream.Write(
                $Buffer,
                0,
                $BytesRead
            )

            $TotalBytes += $BytesRead
        }

        $Stopwatch.Stop()

        $FileStream.Close()
        $Stream.Close()
        $Response.Dispose()
        $Request.Dispose()
        $Content.Dispose()
        $HttpClient.Dispose()

        # 24 kHz / mono / signed 16-bit PCM
        $AudioSeconds = $TotalBytes / (24000 * 1 * 2)

        if ($AudioSeconds -gt 0) {
            $RTF = $Stopwatch.Elapsed.TotalSeconds / $AudioSeconds
        }
        else {
            $RTF = 0
        }

        $Result = [PSCustomObject]@{
            Test              = $Test.Name
            Characters        = $Test.Text.Length
            AudioSeconds      = [math]::Round($AudioSeconds, 2)
            FirstAudioSeconds = [math]::Round($FirstAudioSeconds, 3)
            TotalSeconds      = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 2)
            RTF               = [math]::Round($RTF, 2)
            Bytes             = $TotalBytes
        }

        $PcmResults += $Result

        Write-Host "  Audio:       $($Result.AudioSeconds) sec"
        Write-Host "  First audio: $($Result.FirstAudioSeconds) sec"
        Write-Host "  Total:       $($Result.TotalSeconds) sec"
        Write-Host "  RTF:         $($Result.RTF)x"
        Write-Host ""

    }
    catch {

        $Stopwatch.Stop()

        try {
            $FileStream.Close()
        }
        catch {}

        $HttpClient.Dispose()

        Write-Host "  FAILED" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
    }
}

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

Write-Host ""
Write-Host "========================================="
Write-Host " RESULTS"
Write-Host "========================================="
Write-Host ""

if ($WavResults.Count -gt 0) {

    Write-Host "BUFFERED WAV"
    Write-Host "-----------------------------------------"
    $WavResults | Format-Table -AutoSize
}

Write-Host ""

if ($PcmResults.Count -gt 0) {

    Write-Host "STREAMING PCM"
    Write-Host "-----------------------------------------"
    $PcmResults | Format-Table -AutoSize
}

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

Write-Host ""
Write-Host "========================================="
Write-Host " INTERPRETATION"
Write-Host "========================================="
Write-Host ""

if ($PcmResults.Count -gt 0) {

    foreach ($Result in $PcmResults) {

        if ($Result.FirstAudioSeconds -lt 0.5) {
            $Latency = "Excellent"
        }
        elseif ($Result.FirstAudioSeconds -lt 1.0) {
            $Latency = "Good"
        }
        elseif ($Result.FirstAudioSeconds -lt 2.0) {
            $Latency = "Noticeable"
        }
        else {
            $Latency = "High"
        }

        if ($Result.RTF -lt 0.75) {
            $Speed = "Excellent"
        }
        elseif ($Result.RTF -lt 1.0) {
            $Speed = "Good"
        }
        elseif ($Result.RTF -lt 1.5) {
            $Speed = "Slower than real time"
        }
        else {
            $Speed = "Significantly slower"
        }

        Write-Host "$($Result.Test):"
        Write-Host "  First audio: $($Result.FirstAudioSeconds)s — $Latency"
        Write-Host "  RTF:         $($Result.RTF)x — $Speed"
        Write-Host ""
    }
}

Write-Host "========================================="
Write-Host " BENCHMARK COMPLETE"
Write-Host "========================================="
Write-Host ""

Write-Host "WAV files:"
Write-Host "  $WavDir"
Write-Host ""

Write-Host "PCM files:"
Write-Host "  $PcmDir"
Write-Host ""