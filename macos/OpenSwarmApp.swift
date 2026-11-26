import AppKit
import Foundation
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKUIDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var serverProc: Process?
    private var serverPort: Int = 18790

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMainMenu()

        serverPort = findFreePort(preferred: 18790)
        startPythonServer(port: serverPort)
        waitForHealth(port: serverPort, timeoutSec: 6.0)

        let rect = NSRect(x: 0, y: 0, width: 1320, height: 860)
        window = NSWindow(
            contentRect: rect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Open Swarm"
        window.titlebarAppearsTransparent = true
        window.backgroundColor = NSColor(calibratedRed: 0.984, green: 0.984, blue: 0.988, alpha: 1.0)
        window.center()
        window.appearance = NSAppearance(named: .aqua)
        window.delegate = self

        let config = WKWebViewConfiguration()
        webView = WKWebView(frame: rect, configuration: config)
        webView.uiDelegate = self
        webView.autoresizingMask = [.width, .height]
        window.contentView = webView

        if let url = URL(string: "http://127.0.0.1:\(serverPort)/") {
            webView.load(URLRequest(url: url))
        }

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let p = serverProc, p.isRunning {
            p.terminate()
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping () -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = "Open Swarm"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.runModal()
        completionHandler()
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (Bool) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = "Open Swarm"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        completionHandler(alert.runModal() == .alertFirstButtonReturn)
    }

    @objc private func reloadPage() {
        webView?.reload()
    }

    private func setupMainMenu() {
        let mainMenu = NSMenu()
        let appItem = NSMenuItem()
        mainMenu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Reload", action: #selector(reloadPage), keyEquivalent: "r")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Open Swarm", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        let editItem = NSMenuItem()
        mainMenu.addItem(editItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu

        NSApp.mainMenu = mainMenu
    }

    private func resolveServerScript() -> String? {
        if let envScript = ProcessInfo.processInfo.environment["OPEN_SWARM_SERVER_SCRIPT"],
           !envScript.isEmpty,
           FileManager.default.fileExists(atPath: envScript) {
            return envScript
        }
        if let resPath = Bundle.main.resourcePath {
            let bundled = (resPath as NSString).appendingPathComponent("open_swarm/desktop_server.py")
            if FileManager.default.isExecutableFile(atPath: bundled) || FileManager.default.fileExists(atPath: bundled) {
                return bundled
            }
        }
        let cwdFallback = (FileManager.default.currentDirectoryPath as NSString).appendingPathComponent("desktop_server.py")
        if FileManager.default.fileExists(atPath: cwdFallback) {
            return cwdFallback
        }
        return nil
    }

    private func resolvePythonBinary() -> String {
        for cand in ["/usr/local/bin/python3", "/opt/homebrew/bin/python3", "/usr/bin/python3"] {
            if FileManager.default.isExecutableFile(atPath: cand) {
                return cand
            }
        }
        return "/usr/bin/env"
    }

    private func startPythonServer(port: Int) {
        guard let script = resolveServerScript() else { return }
        let proc = Process()
        let py = resolvePythonBinary()
        proc.executableURL = URL(fileURLWithPath: py)
        if py == "/usr/bin/env" {
            proc.arguments = ["python3", script, "--host", "127.0.0.1", "--port", "\(port)"]
        } else {
            proc.arguments = [script, "--host", "127.0.0.1", "--port", "\(port)"]
        }
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env
        do {
            try proc.run()
            serverProc = proc
        } catch {
            NSLog("Failed to spawn Open Swarm desktop backend: \(error)")
        }
    }

    private func findFreePort(preferred: Int) -> Int {
        for candidate in [preferred, 18791, 18792, 18793] {
            if isPortFree(candidate) { return candidate }
        }
        return 18799
    }

    private func isPortFree(_ port: Int) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        if fd < 0 { return false }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let res = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return res == 0
    }

    private func waitForHealth(port: Int, timeoutSec: Double) {
        let deadline = Date().addingTimeInterval(timeoutSec)
        guard let url = URL(string: "http://127.0.0.1:\(port)/health") else { return }
        while Date() < deadline {
            let sem = DispatchSemaphore(value: 0)
            var ok = false
            let task = URLSession.shared.dataTask(with: url) { _, resp, _ in
                if let http = resp as? HTTPURLResponse, http.statusCode == 200 {
                    ok = true
                }
                sem.signal()
            }
            task.resume()
            _ = sem.wait(timeout: .now() + 0.4)
            if ok { return }
            Thread.sleep(forTimeInterval: 0.1)
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.setActivationPolicy(.regular)
app.delegate = delegate
app.run()
