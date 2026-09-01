#import <Cocoa/Cocoa.h>

@interface VideoSubtitleAppDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, copy) NSString *projectDirectory;
@property(nonatomic, copy) NSString *defaultOutputDirectory;
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) NSTextField *statusLabel;
@property(nonatomic, strong) NSTextField *detailLabel;
@property(nonatomic, strong) NSProgressIndicator *progressBar;
@property(nonatomic, strong) NSTask *task;
@property(nonatomic, strong) NSPipe *pipe;
@property(nonatomic, strong) NSFileHandle *logHandle;
@property(nonatomic, strong) NSMutableString *lineBuffer;
@property(nonatomic, strong) NSMutableArray<NSString *> *outputPaths;
@property(nonatomic, strong) NSMutableArray<NSString *> *errorLines;
@property(nonatomic, strong) NSMutableArray<NSString *> *pendingInputs;
@property(nonatomic) NSInteger currentIndex;
@property(nonatomic) NSInteger totalVideos;
@property(nonatomic, copy) NSString *currentName;
@property(nonatomic) BOOL started;
@end

@implementation VideoSubtitleAppDelegate

- (instancetype)init {
    self = [super init];
    if (self) {
        NSURL *executable = [NSURL fileURLWithPath:NSProcessInfo.processInfo.arguments.firstObject];
        NSURL *resources = [[[executable URLByDeletingLastPathComponent]
            URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"Resources"];
        NSString *projectPath = [[NSString alloc]
            initWithContentsOfURL:[resources URLByAppendingPathComponent:@"project-path"]
            encoding:NSUTF8StringEncoding error:nil];
        _projectDirectory = [projectPath ?: @"" stringByTrimmingCharactersInSet:
            NSCharacterSet.whitespaceAndNewlineCharacterSet];
        _defaultOutputDirectory = [NSHomeDirectory()
            stringByAppendingPathComponent:@"Movies/Video Subtitle Pipeline Outputs"];
        _lineBuffer = [NSMutableString string];
        _outputPaths = [NSMutableArray array];
        _errorLines = [NSMutableArray array];
        _pendingInputs = [NSMutableArray array];
        _totalVideos = 1;
        _currentName = @"";
    }
    return self;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    NSArray<NSString *> *arguments = NSProcessInfo.processInfo.arguments;
    NSFileManager *manager = NSFileManager.defaultManager;
    for (NSUInteger index = 1; index < arguments.count; index++) {
        NSString *candidate = arguments[index];
        if ([manager fileExistsAtPath:candidate]) {
            [self.pendingInputs addObject:candidate];
        }
    }

    __weak typeof(self) weakSelf = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 350 * NSEC_PER_MSEC),
                   dispatch_get_main_queue(), ^{
        typeof(self) self = weakSelf;
        if (!self || self.started) return;
        if (self.pendingInputs.count == 0) {
            [self showUsageAndQuit];
        } else {
            [self begin:self.pendingInputs.copy];
        }
    });
}

- (void)application:(NSApplication *)sender openFiles:(NSArray<NSString *> *)filenames {
    [sender replyToOpenOrPrint:NSApplicationDelegateReplySuccess];
    if (self.started) {
        NSAlert *alert = [[NSAlert alloc] init];
        alert.messageText = @"A subtitle run is already in progress";
        alert.informativeText = @"Wait for the current run to finish, then drag the remaining files onto the app again.";
        [alert runModal];
        return;
    }
    [self.pendingInputs addObjectsFromArray:filenames];
    if (NSApp.running) {
        [self begin:self.pendingInputs.copy];
    }
}

- (void)showUsageAndQuit {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Video Subtitle Pipeline";
    alert.informativeText = @"Drag one or more video files, or a folder containing videos, onto this app.";
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
    [NSApp terminate:nil];
}

- (NSString *)chooseOutputDirectory {
    [NSFileManager.defaultManager createDirectoryAtPath:self.defaultOutputDirectory
        withIntermediateDirectories:YES attributes:nil error:nil];

    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Choose where to save the subtitles";
    alert.informativeText = [NSString stringWithFormat:
        @"The default output folder is:\n%@", self.defaultOutputDirectory];
    [alert addButtonWithTitle:@"Use Default"];
    [alert addButtonWithTitle:@"Choose Folder…"];
    [alert addButtonWithTitle:@"Cancel"];

    NSModalResponse response = [alert runModal];
    if (response == NSAlertFirstButtonReturn) return self.defaultOutputDirectory;
    if (response == NSAlertThirdButtonReturn) return nil;

    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.title = @"Choose Video Subtitle Output Folder";
    panel.prompt = @"Choose";
    panel.canChooseFiles = NO;
    panel.canChooseDirectories = YES;
    panel.allowsMultipleSelection = NO;
    panel.canCreateDirectories = YES;
    panel.directoryURL = [NSURL fileURLWithPath:self.defaultOutputDirectory];
    return [panel runModal] == NSModalResponseOK
        ? panel.URL.path ?: self.defaultOutputDirectory
        : self.defaultOutputDirectory;
}

- (void)begin:(NSArray<NSString *> *)inputs {
    if (self.started) return;
    self.started = YES;
    [self.pendingInputs removeAllObjects];

    NSString *launcher = [self.projectDirectory stringByAppendingPathComponent:@"run-local-drop.sh"];
    if (self.projectDirectory.length == 0 ||
        ![NSFileManager.defaultManager isExecutableFileAtPath:launcher]) {
        [self showImmediateError:@"The project launcher could not be found"
            detail:@"Re-run install-macos-shortcuts.sh from the project directory."];
        return;
    }
    NSString *outputDirectory = [self chooseOutputDirectory];
    if (!outputDirectory) {
        [NSApp terminate:nil];
        return;
    }

    NSError *directoryError = nil;
    if (![NSFileManager.defaultManager createDirectoryAtPath:outputDirectory
          withIntermediateDirectories:YES attributes:nil error:&directoryError]) {
        [self showImmediateError:@"Could not create the output folder"
            detail:directoryError.localizedDescription];
        return;
    }

    [self buildProgressWindow:outputDirectory];
    [self runPipeline:inputs outputDirectory:outputDirectory launcher:launcher];
}

- (void)buildProgressWindow:(NSString *)outputDirectory {
    NSWindow *window = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, 680, 190)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        backing:NSBackingStoreBuffered defer:NO];
    window.title = @"Video Subtitle Pipeline";
    window.releasedWhenClosed = NO;
    [window standardWindowButton:NSWindowCloseButton].enabled = NO;

    NSTextField *status = [NSTextField labelWithString:@"Preparing videos…"];
    status.font = [NSFont systemFontOfSize:16 weight:NSFontWeightSemibold];
    status.lineBreakMode = NSLineBreakByTruncatingMiddle;
    NSTextField *detail = [NSTextField labelWithString:
        [NSString stringWithFormat:@"Saving to: %@", outputDirectory]];
    detail.textColor = NSColor.secondaryLabelColor;
    detail.lineBreakMode = NSLineBreakByTruncatingMiddle;
    NSProgressIndicator *progress = [[NSProgressIndicator alloc] init];
    progress.style = NSProgressIndicatorStyleBar;
    progress.minValue = 0;
    progress.maxValue = 1;
    progress.doubleValue = 0.02;
    progress.indeterminate = NO;
    NSTextField *note = [NSTextField labelWithString:
        @"Keep this window open. Long videos can take several minutes."];
    note.textColor = NSColor.tertiaryLabelColor;

    for (NSView *view in @[status, detail, progress, note]) {
        view.translatesAutoresizingMaskIntoConstraints = NO;
        [window.contentView addSubview:view];
    }
    [NSLayoutConstraint activateConstraints:@[
        [status.leadingAnchor constraintEqualToAnchor:window.contentView.leadingAnchor constant:24],
        [status.trailingAnchor constraintEqualToAnchor:window.contentView.trailingAnchor constant:-24],
        [status.topAnchor constraintEqualToAnchor:window.contentView.topAnchor constant:26],
        [detail.leadingAnchor constraintEqualToAnchor:status.leadingAnchor],
        [detail.trailingAnchor constraintEqualToAnchor:status.trailingAnchor],
        [detail.topAnchor constraintEqualToAnchor:status.bottomAnchor constant:10],
        [progress.leadingAnchor constraintEqualToAnchor:status.leadingAnchor],
        [progress.trailingAnchor constraintEqualToAnchor:status.trailingAnchor],
        [progress.topAnchor constraintEqualToAnchor:detail.bottomAnchor constant:22],
        [progress.heightAnchor constraintEqualToConstant:16],
        [note.leadingAnchor constraintEqualToAnchor:status.leadingAnchor],
        [note.trailingAnchor constraintEqualToAnchor:status.trailingAnchor],
        [note.topAnchor constraintEqualToAnchor:progress.bottomAnchor constant:16]
    ]];

    self.window = window;
    self.statusLabel = status;
    self.detailLabel = detail;
    self.progressBar = progress;
    [window center];
    [window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)runPipeline:(NSArray<NSString *> *)inputs
      outputDirectory:(NSString *)outputDirectory
              launcher:(NSString *)launcher {
    NSString *logPath = [outputDirectory stringByAppendingPathComponent:@"latest-run.log"];
    [NSFileManager.defaultManager createFileAtPath:logPath contents:nil attributes:nil];
    self.logHandle = [NSFileHandle fileHandleForWritingAtPath:logPath];

    NSTask *task = [[NSTask alloc] init];
    NSPipe *pipe = [NSPipe pipe];
    task.executableURL = [NSURL fileURLWithPath:@"/bin/zsh"];
    task.arguments = [@[launcher] arrayByAddingObjectsFromArray:inputs];
    NSMutableDictionary *environment = NSProcessInfo.processInfo.environment.mutableCopy;
    environment[@"VIDEO_SUBTITLE_OUTPUT_DIR"] = outputDirectory;
    task.environment = environment;
    task.standardOutput = pipe;
    task.standardError = pipe;

    __weak typeof(self) weakSelf = self;
    pipe.fileHandleForReading.readabilityHandler = ^(NSFileHandle *handle) {
        NSData *data = handle.availableData;
        if (data.length == 0) {
            handle.readabilityHandler = nil;
            return;
        }
        NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"";
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf consume:text];
        });
    };
    task.terminationHandler = ^(NSTask *finished) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 150 * NSEC_PER_MSEC),
                       dispatch_get_main_queue(), ^{
            [weakSelf finish:finished.terminationStatus
                outputDirectory:outputDirectory logPath:logPath];
        });
    };

    self.task = task;
    self.pipe = pipe;
    NSError *runError = nil;
    if (![task launchAndReturnError:&runError]) {
        [self consume:[NSString stringWithFormat:@"error: %@\n", runError.localizedDescription]];
        [self finish:1 outputDirectory:outputDirectory logPath:logPath];
    }
}

- (void)consume:(NSString *)text {
    NSData *data = [text dataUsingEncoding:NSUTF8StringEncoding];
    if (data) [self.logHandle writeData:data];
    [self.lineBuffer appendString:text];
    while (YES) {
        NSRange newline = [self.lineBuffer rangeOfString:@"\n"];
        if (newline.location == NSNotFound) break;
        NSString *line = [self.lineBuffer substringToIndex:newline.location];
        [self.lineBuffer deleteCharactersInRange:NSMakeRange(0, NSMaxRange(newline))];
        [self handleLine:line];
    }
}

- (void)handleLine:(NSString *)line {
    NSRegularExpression *pattern = [NSRegularExpression
        regularExpressionWithPattern:@"^\\[(\\d+)/(\\d+)\\] Processing: (.+)$"
        options:0 error:nil];
    NSTextCheckingResult *match = [pattern firstMatchInString:line options:0
        range:NSMakeRange(0, line.length)];
    if (match) {
        self.currentIndex = [[line substringWithRange:[match rangeAtIndex:1]] integerValue];
        self.totalVideos = MAX([[line substringWithRange:[match rangeAtIndex:2]] integerValue], 1);
        NSString *path = [line substringWithRange:[match rangeAtIndex:3]];
        self.currentName = path.lastPathComponent;
        [self updateProgress:0.05 description:@"Preparing"];
    } else if ([line hasPrefix:@"ASR:"]) {
        [self updateProgress:0.18 description:@"Local Metal speech recognition"];
    } else if ([line hasPrefix:@"translate-batch-"]) {
        [self updateProgress:0.48 description:@"Korean translation with Codex"];
    } else if ([line hasPrefix:@"Output #0"]) {
        [self updateProgress:0.72 description:@"Rendering hard subtitles"];
    } else if ([line hasPrefix:@"Wrote video:"]) {
        [self updateProgress:0.96 description:@"Finishing output"];
        [self captureOutput:line];
    } else if ([line hasPrefix:@"Wrote subtitles:"] || [line hasPrefix:@"Wrote manifest:"]) {
        [self captureOutput:line];
    }

    if ([line rangeOfString:@"error:" options:NSCaseInsensitiveSearch].location != NSNotFound) {
        [self.errorLines addObject:line];
        if (self.errorLines.count > 8) [self.errorLines removeObjectAtIndex:0];
    }
}

- (void)captureOutput:(NSString *)line {
    NSRange separator = [line rangeOfString:@":"];
    if (separator.location == NSNotFound) return;
    NSString *path = [[line substringFromIndex:NSMaxRange(separator)]
        stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
    if (path.length > 0 && ![self.outputPaths containsObject:path]) {
        [self.outputPaths addObject:path];
    }
}

- (void)updateProgress:(double)phase description:(NSString *)description {
    double total = MAX(self.totalVideos, 1);
    double completed = MAX(self.currentIndex - 1, 0);
    self.progressBar.maxValue = total;
    self.progressBar.doubleValue = MIN(completed + phase, total);
    NSString *name = self.currentName.length ? self.currentName : @"video";
    self.statusLabel.stringValue = [NSString stringWithFormat:@"%@ — %@", description, name];
    self.detailLabel.stringValue = [NSString stringWithFormat:@"Video %ld of %ld",
        (long)MAX(self.currentIndex, 1), (long)self.totalVideos];
}

- (void)finish:(int)status outputDirectory:(NSString *)outputDirectory logPath:(NSString *)logPath {
    self.pipe.fileHandleForReading.readabilityHandler = nil;
    if (self.lineBuffer.length > 0) {
        [self handleLine:self.lineBuffer.copy];
        [self.lineBuffer setString:@""];
    }
    [self.logHandle closeFile];
    self.progressBar.doubleValue = self.progressBar.maxValue;
    [self.window orderOut:nil];

    BOOL succeeded = status == 0;
    NSAlert *alert = [[NSAlert alloc] init];
    alert.alertStyle = succeeded ? NSAlertStyleInformational : NSAlertStyleCritical;
    alert.messageText = succeeded ? @"Video subtitles completed" : @"Video subtitle processing failed";
    NSMutableString *details = [NSMutableString stringWithFormat:
        @"Output folder:\n%@", outputDirectory];
    if (self.outputPaths.count > 0) {
        [details appendString:@"\n\nCreated files:\n"];
        NSUInteger visibleCount = MIN(self.outputPaths.count, 15);
        [details appendString:[[self.outputPaths subarrayWithRange:NSMakeRange(0, visibleCount)]
            componentsJoinedByString:@"\n"]];
        if (self.outputPaths.count > visibleCount) {
            [details appendFormat:@"\n…and %lu more files",
                (unsigned long)(self.outputPaths.count - visibleCount)];
        }
    }
    if (!succeeded) {
        [details appendFormat:@"\n\nLog:\n%@", logPath];
        if (self.errorLines.count > 0) {
            [details appendFormat:@"\n\nLast error:\n%@", self.errorLines.lastObject];
        }
    }
    alert.informativeText = details;
    NSString *videoPath = nil;
    for (NSString *outputPath in self.outputPaths.reverseObjectEnumerator) {
        if ([outputPath.pathExtension caseInsensitiveCompare:@"mp4"] == NSOrderedSame) {
            videoPath = outputPath;
            break;
        }
    }

    if (succeeded) {
        [alert addButtonWithTitle:self.totalVideos > 1 ? @"Play First Video" : @"Play Video"];
        [alert addButtonWithTitle:@"Show in Finder"];
        [alert addButtonWithTitle:@"Close"];
        NSButton *playButton = alert.buttons.firstObject;
        playButton.enabled = videoPath.length > 0 &&
            [NSFileManager.defaultManager fileExistsAtPath:videoPath];
        NSModalResponse response = [alert runModal];
        if (response == NSAlertFirstButtonReturn && playButton.enabled) {
            [NSWorkspace.sharedWorkspace openURL:[NSURL fileURLWithPath:videoPath]];
        } else if (response == NSAlertSecondButtonReturn) {
            if (videoPath.length > 0) {
                [NSWorkspace.sharedWorkspace selectFile:videoPath
                    inFileViewerRootedAtPath:outputDirectory];
            } else {
                [NSWorkspace.sharedWorkspace openURL:[NSURL fileURLWithPath:outputDirectory]];
            }
        }
    } else {
        [alert addButtonWithTitle:@"Open Log"];
        [alert addButtonWithTitle:@"Show Output Folder"];
        [alert addButtonWithTitle:@"Close"];
        NSModalResponse response = [alert runModal];
        if (response == NSAlertFirstButtonReturn) {
            NSURL *logURL = [NSURL fileURLWithPath:logPath];
            if (![NSWorkspace.sharedWorkspace openURL:logURL]) {
                [NSWorkspace.sharedWorkspace selectFile:logPath
                    inFileViewerRootedAtPath:outputDirectory];
            }
        } else if (response == NSAlertSecondButtonReturn) {
            [NSWorkspace.sharedWorkspace openURL:[NSURL fileURLWithPath:outputDirectory]];
        }
    }
    [NSApp terminate:nil];
}

- (void)showImmediateError:(NSString *)message detail:(NSString *)detail {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.alertStyle = NSAlertStyleCritical;
    alert.messageText = message;
    alert.informativeText = detail;
    [alert runModal];
    [NSApp terminate:nil];
}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *application = NSApplication.sharedApplication;
        VideoSubtitleAppDelegate *delegate = [[VideoSubtitleAppDelegate alloc] init];
        application.delegate = delegate;
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
        [application run];
    }
    return 0;
}
