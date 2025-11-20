#!/usr/bin/env ruby
# Script to integrate ONNX Runtime into Xcode project

require 'xcodeproj'

project_path = 'mels.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first

# Add the dylib
lib_path = 'onnxruntime-osx-universal2-1.16.3/lib/libonnxruntime.1.16.3.dylib'
file_ref = project.new_file(lib_path)
target.frameworks_build_phase.add_file_reference(file_ref)

# Add header search path
target.build_configurations.each do |config|
  config.build_settings['HEADER_SEARCH_PATHS'] ||= ['$(inherited)']
  config.build_settings['HEADER_SEARCH_PATHS'] << '$(PROJECT_DIR)/onnxruntime-osx-universal2-1.16.3/include'
  
  config.build_settings['LIBRARY_SEARCH_PATHS'] ||= ['$(inherited)']
  config.build_settings['LIBRARY_SEARCH_PATHS'] << '$(PROJECT_DIR)/onnxruntime-osx-universal2-1.16.3/lib'
  
  config.build_settings['SWIFT_OBJC_BRIDGING_HEADER'] = 'mels/mels-Bridging-Header.h'
end

project.save

puts "✓ ONNX Runtime integrated into Xcode project"

