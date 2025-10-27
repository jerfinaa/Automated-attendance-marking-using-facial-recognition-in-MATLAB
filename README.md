classdef AttendanceApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        GridLayout                      matlab.ui.container.GridLayout
        LeftPanel                       matlab.ui.container.Panel
        LoadClassListButton             matlab.ui.control.Button
        CreateFaceDatabaseButton        matlab.ui.control.Button
        StartCameraButton               matlab.ui.control.Button
        CaptureMarkAttendanceButton     matlab.ui.control.Button
        SaveAttendanceButton            matlab.ui.control.Button
        StatusLabel                     matlab.ui.control.Label
        RecognitionThresholdSpinnerLabel  matlab.ui.control.Label
        RecognitionThresholdSpinner     matlab.ui.control.Spinner
        KNNSpinnerLabel                 matlab.ui.control.Label
        KNNSpinner                      matlab.ui.control.Spinner
        RightPanel                      matlab.ui.container.Panel
        AttendanceUITable               matlab.ui.control.Table
        CapturedImageAxes               matlab.ui.control.UIAxes
        CameraFeedAxes                  matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        ClassData % Table to store student registration numbers and names
        WebcamObj % Webcam object
        FaceDetector % Vision Cascade Object Detector for faces (Slow, Accurate - for Database)
        
        % --- PERFORMANCE UPGRADE: Add a faster detector for live capture ---
        LiveFaceDetector % Faster LBP detector for live camera feed
        % --- End of Upgrade ---

        % --- Properties for Facial Recognition ---
        FaceNet % Pre-trained deep learning network for feature extraction
        
        % --- ACCURACY UPGRADE: Properties for KNN matching ---
        FaceFeatures % N x 2048 matrix of all features from all database images
        FaceLabels   % N x 1 cell array of RegNo labels for each feature
        % --- End of Upgrade ---

        DatabaseLoaded (1,1) logical = false
        % Default recognition threshold
        RecognitionThreshold (1,1) double = 1.4;
        % --- SCALABILITY UPGRADE: K for KNN ---
        KNN_K (1,1) double = 5; % Number of neighbors to check
        % --- End of Upgrade ---
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            app.UIFigure.Name = "Automated Attendance System (Face Recognition)";
            
            % --- DEBUGGING: Use more sensitive (CART) detector ---
            % This (CART) is for the database creation step, where
            % accuracy is most important.
            try
                app.FaceDetector = vision.CascadeObjectDetector('MinSize', [40 40]);
            catch
                uialert(app.UIFigure, 'Failed to create face detector (CART). Is the Computer Vision Toolbox installed?', 'Detector Error');
                app.CaptureMarkAttendanceButton.Enable = 'off';
                app.CreateFaceDatabaseButton.Enable = 'off';
            end
            
            % --- PERFORMANCE UPGRADE: Use faster (LBP) detector for live capture ---
            % This detector is much faster than CART and is suitable for
            % real-time camera processing.
            try
                app.LiveFaceDetector = vision.CascadeObjectDetector('FrontalFaceLBP', 'MinSize', [80 80]);
            catch ME
                % If LBP fails, fall back to the main detector
                warning('LBP Detector failed to load. Falling back to CART. Performance may be slow. Error: %s', ME.message);
                app.LiveFaceDetector = app.FaceDetector;
            end
            % --- End of Change ---
            
            % Load the pre-trained network
            try
                app.FaceNet = resnet50;
                app.StatusLabel.Text = "Status: Ready. Please load class list.";
            catch ME
                uialert(app.UIFigure, 'Could not load ResNet-50. Please install the Deep Learning Toolbox Model for ResNet-50 Network.', 'Network Error');
                app.StatusLabel.Text = "Error: Deep Learning model missing.";
                app.CaptureMarkAttendanceButton.Enable = 'off';
                app.CreateFaceDatabaseButton.Enable = 'off';
            end
            
            % Set initial spinner values from properties
            app.RecognitionThresholdSpinner.Value = app.RecognitionThreshold;
            app.KNNSpinner.Value = app.KNN_K;
            
            % Initialize new database properties
            app.FaceFeatures = [];
            app.FaceLabels = {};
            
            % Disable buttons until prerequisites are met
            app.CreateFaceDatabaseButton.Enable = 'off';
            app.StartCameraButton.Enable = 'off';
            app.CaptureMarkAttendanceButton.Enable = 'off';
            app.SaveAttendanceButton.Enable = 'off';
        end

        % Button pushed function: LoadClassListButton
        function LoadClassListButtonPushed(app, event)
            [file, path] = uigetfile('*.xlsx', 'Select the Class List Excel File');
            if isequal(file, 0), return; end
            
            try
                filePath = fullfile(path, file);
                opts = detectImportOptions(filePath);
                opts.SelectedVariableNames = {'RegNo', 'Name'}; 
                app.ClassData = readtable(filePath, opts);
                app.ClassData.Status = repmat({'Absent'}, height(app.ClassData), 1); 
                app.AttendanceUITable.Data = app.ClassData;
                
                app.StatusLabel.Text = "Status: Class list loaded. Please create face database.";
                app.CreateFaceDatabaseButton.Enable = 'on';
                app.SaveAttendanceButton.Enable = 'on';
                removeStyle(app.AttendanceUITable); 
            catch ME
                uialert(app.UIFigure, ['Failed to read Excel file. Error: ' ME.message], 'File Error');
            end
        end

        % Button pushed function: CreateFaceDatabaseButton
        function CreateFaceDatabaseButtonPushed(app, event)
            [zipFile, zipPath] = uigetfile('*.zip', 'Select the Student Images Zip File');
            if isequal(zipFile, 0), return; end
            
            app.StatusLabel.Text = "Status: Unzipping student images...";
            drawnow;

            try
                outputDir = 'Extracted_Student_Images';
                if exist(outputDir, 'dir')
                    rmdir(outputDir, 's');
                end
                mkdir(outputDir);
                unzip(fullfile(zipPath, zipFile), outputDir);
                
                unzippedContents = dir(outputDir);
                if length(unzippedContents) == 3 && unzippedContents(3).isdir 
                    dbPath = fullfile(outputDir, unzippedContents(3).name);
                else
                    dbPath = outputDir; 
                end

                app.StatusLabel.Text = "Status: Creating face database... This may take a moment.";
                drawnow; 

                studentFolders = dir(dbPath);
                studentFolders = studentFolders([studentFolders.isdir] & ~ismember({studentFolders.name},{'.','..'}));

                if isempty(studentFolders)
                    uialert(app.UIFigure, 'No student folders found in the extracted directory.', 'Database Error');
                    return;
                end
                
                % --- Initialize KNN database ---
                app.FaceFeatures = [];
                app.FaceLabels = {};
                
                netInputSize = app.FaceNet.Layers(1).InputSize(1:2);
                featureLayer = 'avg_pool';
                
                for i = 1:length(studentFolders)
                    regNo = string(studentFolders(i).name);
                    
                    nameIdx = strcmpi(app.ClassData.RegNo, regNo);
                    if ~any(nameIdx)
                        warning('Folder %s does not correspond to any RegNo in the class list. Skipping.', regNo);
                        continue;
                    end
                    studentName = app.ClassData.Name{nameIdx};
                    
                    app.StatusLabel.Text = sprintf("Processing: %s (%d/%d)", studentName, i, length(studentFolders));
                    drawnow;
                    
                    imgFolder = fullfile(dbPath, regNo);
                    imgFiles = imageDatastore(imgFolder, 'IncludeSubfolders', true, 'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp'});
                    
                    if isempty(imgFiles.Files)
                        warning('No valid image files found for student %s in folder %s. Skipping.', regNo, imgFolder);
                        continue; 
                    end
                    
                    % --- Store all features, not the average ---
                    while hasdata(imgFiles)
                        img = read(imgFiles);
                        % --- Use the ACCURATE (CART) detector for database ---
                        bboxes = app.FaceDetector.step(img);
                        
                        if ~isempty(bboxes)
                            [~, maxIdx] = max(bboxes(:,3).*bboxes(:,4));
                            face = imcrop(img, bboxes(maxIdx,:));
                            
                            faceHSV = rgb2hsv(face);
                            faceHSV(:,:,3) = histeq(faceHSV(:,:,3)); 
                            faceNormalized = hsv2rgb(faceHSV);
                            
                            faceResized = imresize(faceNormalized, netInputSize);
                            features = activations(app.FaceNet, faceResized, featureLayer, 'OutputAs', 'rows');
                            
                            app.FaceFeatures = [app.FaceFeatures; features];
                            app.FaceLabels = [app.FaceLabels; {char(regNo)}];
                        end
                    end
                end

                if isempty(app.FaceFeatures)
                    uialert(app.UIFigure, 'Database creation failed. No valid faces were processed.', 'Database Error');
                    return;
                end
                
                % Ensure K is not larger than the number of database samples
                if app.KNN_K > length(app.FaceLabels)
                    app.KNN_K = length(app.FaceLabels);
                    app.KNNSpinner.Value = app.KNN_K;
                    warning('K was larger than total number of photos. Setting K to %d.', app.KNN_K);
                end
                
                app.DatabaseLoaded = true;
                app.StartCameraButton.Enable = 'on';
                app.StatusLabel.Text = "Status: Face database created successfully. Ready for camera.";

            catch ME
                uialert(app.UIFigure, ['Error building database: ' ME.message], 'Database Error');
                app.StatusLabel.Text = "Status: Error building database.";
            end
        end

        % Button pushed function: StartCameraButton
        function StartCameraButtonPushed(app, event)
            try
                if isempty(webcamlist), uialert(app.UIFigure, 'No webcam detected.', 'Webcam Error'); return; end
                if ~isempty(app.WebcamObj), clear app.WebcamObj; end
                app.WebcamObj = webcam;
                preview(app.WebcamObj, app.CameraFeedAxes);
                app.StatusLabel.Text = "Status: Camera started. Ready to capture.";
                app.CaptureMarkAttendanceButton.Enable = 'on';
            catch ME
                uialert(app.UIFigure, ['Could not start webcam. Error: ' ME.message], 'Webcam Error');
                if ~isempty(app.WebcamObj), clear app.WebcamObj; end
            end
        end

        % Button pushed function: CaptureMarkAttendanceButton
        function CaptureMarkAttendanceButtonPushed(app, event)
            if isempty(app.WebcamObj), uialert(app.UIFigure, 'Camera not started.', 'Capture Error'); return; end
            if ~app.DatabaseLoaded, uialert(app.UIFigure, 'Face database not created.', 'Capture Error'); return; end

            app.StatusLabel.Text = "Status: Capturing... Recognizing faces...";
            img = snapshot(app.WebcamObj);
            
            % --- PERFORMANCE UPGRADE: Use the FASTER (LBP) detector for live capture ---
            bboxes = app.LiveFaceDetector.step(img);
            
            if isempty(bboxes)
                app.StatusLabel.Text = "Status: No faces detected.";
                imshow(img, 'Parent', app.CapturedImageAxes);
                title(app.CapturedImageAxes, 'Captured Image (0 faces detected)');
                return;
            end
            
            % Find the largest face by area
            areas = bboxes(:,3) .* bboxes(:,4);
            [~, maxIdx] = max(areas);
            mainBBox = bboxes(maxIdx, :);

            % Process only the largest face
            netInputSize = app.FaceNet.Layers(1).InputSize(1:2);
            featureLayer = 'avg_pool';
            dbFeatures = app.FaceFeatures; 
            
            face = imcrop(img, mainBBox);
            
            % Apply SAME Pre-processing
            faceHSV = rgb2hsv(face);
            faceHSV(:,:,3) = histeq(faceHSV(:,:,3));
            faceNormalized = hsv2rgb(faceHSV);
            faceResized = imresize(faceNormalized, netInputSize);
            
            queryFeatures = activations(app.FaceNet, faceResized, featureLayer, 'OutputAs', 'rows');
            
            % --- SCALABILITY UPGRADE: Robust KNN Matching ---
            distances = pdist2(queryFeatures, dbFeatures, 'euclidean');
            
            % Get the K closest distances and their indices
            [sortedDist, sortedIndices] = sort(distances, 'ascend');
            k_Indices = sortedIndices(1:app.KNN_K);
            k_Distances = sortedDist(1:app.KNN_K);
            
            % Get the labels (RegNo) for these K neighbors
            k_Labels = app.FaceLabels(k_Indices);
            
            % Find the most frequent label (majority vote)
            % --- FIX: Replaced 'mode' with manual voting for compatibility ---
            [unique_labels, ~, label_indices] = unique(k_Labels);
            counts = accumarray(label_indices, 1);
            [majorityCount, max_idx] = max(counts);
            matchedRegNoStr = unique_labels{max_idx};
            matchedRegNo = string(matchedRegNoStr);
            
            % Check if this majority is strong enough
            % 'majorityCount' is already calculated by max(counts)
            % --- End of fix ---
            
            isMajority = majorityCount >= ceil(app.KNN_K / 2); 
            
            % Calculate the average distance of *only* the matching majority faces
            avgDist = mean(k_Distances(strcmp(k_Labels, matchedRegNoStr)));
            
            imgWithBoxes = img; 
            
            if isMajority && (avgDist < app.RecognitionThreshold)
                % --- We have a confident match ---
                studentIdx = strcmpi(app.ClassData.RegNo, matchedRegNo);
                
                % Check if student was found in class list
                if any(studentIdx)
                    matchedName = app.ClassData.Name{studentIdx};
                    app.ClassData.Status{studentIdx} = 'Present';
                    
                    label = char(matchedName);
                    color = 'green';
                    statusText = sprintf("Recognized and marked '%s' as Present.", label);
                else
                    % This should not happen if database is built from class list
                    label = 'Error: Matched RegNo not in class list';
                    color = 'yellow';
                    statusText = "Error: Match not in class list.";
                end
            else
                % --- No confident match ---
                % Find the single closest match just for debugging info
                [minDist, ~] = min(distances); 
                label = sprintf('Unknown (Dist: %.2f)', minDist);
                color = 'red';
                statusText = "Largest face is Unknown.";
            end
            % --- End of Upgrade ---

            imgWithBoxes = insertObjectAnnotation(imgWithBoxes, 'rectangle', mainBBox, label, ...
                'FontSize', 14, 'LineWidth', 4, 'Color', color);
            
            imshow(imgWithBoxes, 'Parent', app.CapturedImageAxes);
            title(app.CapturedImageAxes, 'Captured Image & Recognition Result');
            app.StatusLabel.Text = ['Status: ', statusText];

            app.AttendanceUITable.Data = app.ClassData;
            app.updateTableStyle();
        end
        
        % Helper function to style the table
        function updateTableStyle(app)
            removeStyle(app.AttendanceUITable);
            presentRows = strcmp(app.AttendanceUITable.Data.Status, 'Present');
            if any(presentRows)
                s = uistyle('BackgroundColor', [0.7, 1, 0.7]); % Light green
                addStyle(app.AttendanceUITable, s, 'row', find(presentRows));
            end
        end

        % Value changed function: RecognitionThresholdSpinner
        function RecognitionThresholdSpinnerValueChanged(app, event)
            app.RecognitionThreshold = app.RecognitionThresholdSpinner.Value;
            app.StatusLabel.Text = sprintf('Status: Threshold set to %.2f', app.RecognitionThreshold);
        end

        % --- SCALABILITY UPGRADE: Callback for KNN Spinner ---
        % Value changed function: KNNSpinner
        function KNNSpinnerValueChanged(app, event)
            newK = app.KNNSpinner.Value;
            % Ensure K is not larger than the number of database samples
            if app.DatabaseLoaded && (newK > length(app.FaceLabels))
                newK = length(app.FaceLabels);
                app.KNNSpinner.Value = newK;
                warning('K cannot be larger than total number of photos. Setting K to %d.', newK);
            end
            app.KNN_K = newK;
            app.StatusLabel.Text = sprintf('Status: K (Neighbors) set to %d', app.KNN_K);
        end
        % --- End of Upgrade ---

        % Button pushed function: SaveAttendanceButton
        function SaveAttendanceButtonPushed(app, event)
            if isempty(app.ClassData), uialert(app.UIFigure, 'No data to save.', 'Save Error'); return; end
            try
                timestamp = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
                filename = ['Attendance_', timestamp, '.xlsx'];
                writetable(app.AttendanceUITable.Data, filename);
                msgbox(['Attendance saved as ' filename], 'Success');
            catch ME
                 uialert(app.UIFigure, ['Failed to save file. Error: ' ME.message], 'Save Error');
            end
        end

        % Close request function: UIFigure
        function UIFigureCloseRequest(app, event)
            if ~isempty(app.WebcamObj), clear app.WebcamObj; end
            delete(app);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1021 615];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {220, '1x'};
            app.GridLayout.RowHeight = {'1x'};

            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Title = 'Controls';
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            app.LoadClassListButton = uibutton(app.LeftPanel, 'push');
            app.LoadClassListButton.ButtonPushedFcn = createCallbackFcn(app, @LoadClassListButtonPushed, true);
            app.LoadClassListButton.FontSize = 14;
            app.LoadClassListButton.Text = '1. Load Class List';
            app.LoadClassListButton.Position = [28 535 165 35];

            app.CreateFaceDatabaseButton = uibutton(app.LeftPanel, 'push');
            app.CreateFaceDatabaseButton.ButtonPushedFcn = createCallbackFcn(app, @CreateFaceDatabaseButtonPushed, true);
            app.CreateFaceDatabaseButton.FontSize = 14;
            app.CreateFaceDatabaseButton.Text = '2. Create Face Database';
            app.CreateFaceDatabaseButton.Position = [28 490 165 35];

            app.StartCameraButton = uibutton(app.LeftPanel, 'push');
            app.StartCameraButton.ButtonPushedFcn = createCallbackFcn(app, @StartCameraButtonPushed, true);
            app.StartCameraButton.FontSize = 14;
            app.StartCameraButton.Text = '3. Start Camera';
            app.StartCameraButton.Position = [28 445 165 35];

            app.CaptureMarkAttendanceButton = uibutton(app.LeftPanel, 'push');
            app.CaptureMarkAttendanceButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureMarkAttendanceButtonPushed, true);
            app.CaptureMarkAttendanceButton.FontSize = 14;
            app.CaptureMarkAttendanceButton.Text = '4. Capture & Mark';
            app.CaptureMarkAttendanceButton.Position = [28 400 165 35];

            app.SaveAttendanceButton = uibutton(app.LeftPanel, 'push');
            app.SaveAttendanceButton.ButtonPushedFcn = createCallbackFcn(app, @SaveAttendanceButtonPushed, true);
            app.SaveAttendanceButton.FontSize = 14;
            app.SaveAttendanceButton.Text = '5. Save Attendance';
            app.SaveAttendanceButton.Position = [28 355 165 35];

            % --- SCALABILITY UPGRADE: Add KNN Spinner UI ---
            app.RecognitionThresholdSpinnerLabel = uilabel(app.LeftPanel);
            app.RecognitionThresholdSpinnerLabel.HorizontalAlignment = 'right';
            app.RecognitionThresholdSpinnerLabel.Position = [28 316 128 22];
            app.RecognitionThresholdSpinnerLabel.Text = 'Recognition Threshold';

            app.RecognitionThresholdSpinner = uispinner(app.LeftPanel);
            app.RecognitionThresholdSpinner.Step = 0.05;
            app.RecognitionThresholdSpinner.Limits = [0.5 2.5];
            app.RecognitionThresholdSpinner.ValueChangedFcn = createCallbackFcn(app, @RecognitionThresholdSpinnerValueChanged, true);
            app.RecognitionThresholdSpinner.Position = [168 316 45 22];
            app.RecognitionThresholdSpinner.Value = 1.4;

            app.KNNSpinnerLabel = uilabel(app.LeftPanel);
            app.KNNSpinnerLabel.HorizontalAlignment = 'right';
            app.KNNSpinnerLabel.Position = [28 284 128 22];
            app.KNNSpinnerLabel.Text = 'K (Neighbors)';

            app.KNNSpinner = uispinner(app.LeftPanel);
            app.KNNSpinner.Step = 1;
            app.KNNSpinner.Limits = [1 10];
            app.KNNSpinner.ValueChangedFcn = createCallbackFcn(app, @KNNSpinnerValueChanged, true);
            app.KNNSpinner.Position = [168 284 45 22];
            app.KNNSpinner.Value = 5;
            % --- End of Upgrade ---

            app.StatusLabel = uilabel(app.LeftPanel);
            app.StatusLabel.BackgroundColor = [0.902 0.902 0.902];
            app.StatusLabel.WordWrap = 'on';
            app.StatusLabel.FontSize = 12;
            app.StatusLabel.Position = [28 21 165 142];
            app.StatusLabel.Text = 'Status:';

            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            app.AttendanceUITable = uitable(app.RightPanel);
            app.AttendanceUITable.ColumnName = {'RegNo'; 'Name'; 'Status'};
            app.AttendanceUITable.RowName = {};
            app.AttendanceUITable.FontSize = 12;
            app.AttendanceUITable.Position = [456 16 345 561];

            app.CapturedImageAxes = uiaxes(app.RightPanel);
            title(app.CapturedImageAxes, 'Captured Image')
            app.CapturedImageAxes.XTick = [];
            app.CapturedImageAxes.YTick = [];
            app.CapturedImageAxes.Position = [23 16 414 270];

            app.CameraFeedAxes = uiaxes(app.RightPanel);
            title(app.CameraFeedAxes, 'Camera Feed')
            app.CameraFeedAxes.XTick = [];
            app.CameraFeedAxes.YTick = [];
            app.CameraFeedAxes.Position = [23 306 414 270];

            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = AttendanceApp
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)
            delete(app.UIFigure)
        end
    end
end



