# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Capacitor & WebView Rules
-keep class * extends com.getcapacitor.Plugin
-keepclassmembers class * extends com.getcapacitor.Plugin {
    @com.getcapacitor.annotation.CapacitorPlugin.Method *;
}
-keepattributes *Annotation*
-keepattributes JavascriptInterface
-keep class * implements com.getcapacitor.Bridge { *; }

# Keep WebView JavaScript Interface
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# AndroidX and Material Components
-keep class com.google.android.material.** { *; }
-dontwarn com.google.android.material.**
-dontnote com.google.android.material.**
-keep class androidx.** { *; }
-dontwarn androidx.**

# Preserve line numbers for debugging
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Keep all JSON model classes (if you have any)
-keepclassmembers class * {
    @com.google.gson.annotations.SerializedName <fields>;
}

# Don't warn about missing classes
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**
