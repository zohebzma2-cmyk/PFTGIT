import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import Svg, { Path, Circle } from 'react-native-svg';
import { useAuthStore } from '../../store/authStore';
import { useEditorStore } from '../../store/editorStore';
import { useDeviceStore, ConnectedDevice } from '../../store/deviceStore';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { DeviceSelector } from '../../components/DeviceSelector';
import { getDeviceFeatureDescription } from '../../data/deviceDatabase';
import colors from '../../constants/colors';

// Icons
function UserIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <Circle cx={12} cy={7} r={4} />
    </Svg>
  );
}

function SliderIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" />
    </Svg>
  );
}

function BluetoothIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M6.5 6.5l11 11L12 23V1l5.5 5.5-11 11" />
    </Svg>
  );
}

function InfoIcon({ color = colors.text.primary, size = 24 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Circle cx={12} cy={12} r={10} />
      <Path d="M12 16v-4M12 8h.01" />
    </Svg>
  );
}

function ChevronRightIcon({ color = colors.text.muted, size = 20 }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}>
      <Path d="M9 18l6-6-6-6" />
    </Svg>
  );
}

interface SettingRowProps {
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
  onPress?: () => void;
  rightElement?: React.ReactNode;
}

function SettingRow({ icon, title, subtitle, onPress, rightElement }: SettingRowProps) {
  const content = (
    <View style={styles.settingRow}>
      <View style={styles.settingIcon}>{icon}</View>
      <View style={styles.settingContent}>
        <Text style={styles.settingTitle}>{title}</Text>
        {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
      </View>
      {rightElement || (onPress && <ChevronRightIcon />)}
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        {content}
      </TouchableOpacity>
    );
  }

  return content;
}

export default function SettingsScreen() {
  const router = useRouter();
  const { user, isAuthenticated, logout } = useAuthStore();
  const { settings, setSettings } = useEditorStore();
  const { connectedDevice, setConnectedDevice, disconnect } = useDeviceStore();
  const [deviceSelectorVisible, setDeviceSelectorVisible] = useState(false);

  const handleDeviceConnected = (device: ConnectedDevice) => {
    setConnectedDevice(device);
  };

  const handleDisconnectDevice = () => {
    Alert.alert(
      'Disconnect Device',
      `Are you sure you want to disconnect ${connectedDevice?.name}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Disconnect',
          style: 'destructive',
          onPress: () => disconnect(),
        },
      ]
    );
  };

  const handleLogout = () => {
    Alert.alert('Log Out', 'Are you sure you want to log out?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Log Out',
        style: 'destructive',
        onPress: async () => {
          await logout();
          router.replace('/(auth)/login');
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
        {/* Account Section */}
        <Card style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          {isAuthenticated && user ? (
            <>
              <SettingRow
                icon={<UserIcon color={colors.primary.DEFAULT} />}
                title={user.username}
                subtitle={user.email}
              />
              <Button
                title="Log Out"
                variant="outline"
                onPress={handleLogout}
                style={styles.logoutButton}
              />
            </>
          ) : (
            <View style={styles.authButtons}>
              <Button
                title="Sign In"
                onPress={() => router.push('/(auth)/login')}
                fullWidth
              />
              <Button
                title="Create Account"
                variant="outline"
                onPress={() => router.push('/(auth)/register')}
                fullWidth
              />
            </View>
          )}
        </Card>

        {/* AI Processing Section */}
        <Card style={styles.section}>
          <Text style={styles.sectionTitle}>AI Processing</Text>

          <SettingRow
            icon={<SliderIcon color={colors.primary.DEFAULT} />}
            title="Confidence Threshold"
            subtitle={`${Math.round(settings.confidenceThreshold * 100)}%`}
          />

          <SettingRow
            icon={<SliderIcon color={colors.primary.DEFAULT} />}
            title="Smoothing Factor"
            subtitle={`${Math.round(settings.smoothingFactor * 100)}%`}
          />

          <SettingRow
            icon={<SliderIcon color={colors.primary.DEFAULT} />}
            title="Min Point Distance"
            subtitle={`${settings.minPointDistance}ms`}
          />

          <SettingRow
            icon={<SliderIcon color={colors.primary.DEFAULT} />}
            title="Auto-smooth Peaks"
            rightElement={
              <Switch
                value={settings.autoSmoothPeaks}
                onValueChange={(value) => setSettings({ autoSmoothPeaks: value })}
                trackColor={{ false: colors.bg.highlight, true: colors.primary.dim }}
                thumbColor={settings.autoSmoothPeaks ? colors.primary.DEFAULT : colors.text.muted}
              />
            }
          />

          <SettingRow
            icon={<SliderIcon color={colors.primary.DEFAULT} />}
            title="Invert Output"
            rightElement={
              <Switch
                value={settings.invertOutput}
                onValueChange={(value) => setSettings({ invertOutput: value })}
                trackColor={{ false: colors.bg.highlight, true: colors.primary.dim }}
                thumbColor={settings.invertOutput ? colors.primary.DEFAULT : colors.text.muted}
              />
            }
          />

          <SettingRow
            icon={<SliderIcon color={colors.primary.DEFAULT} />}
            title="Limit Range (0-100)"
            rightElement={
              <Switch
                value={settings.limitRange}
                onValueChange={(value) => setSettings({ limitRange: value })}
                trackColor={{ false: colors.bg.highlight, true: colors.primary.dim }}
                thumbColor={settings.limitRange ? colors.primary.DEFAULT : colors.text.muted}
              />
            }
          />
        </Card>

        {/* Device Section */}
        <Card style={styles.section}>
          <Text style={styles.sectionTitle}>Device</Text>

          {connectedDevice ? (
            <>
              <SettingRow
                icon={<BluetoothIcon color={colors.status.success} />}
                title={connectedDevice.name}
                subtitle={`${connectedDevice.manufacturer} • ${getDeviceFeatureDescription(connectedDevice.features)}`}
                rightElement={
                  <View style={styles.connectedBadge}>
                    <Text style={styles.connectedBadgeText}>Connected</Text>
                  </View>
                }
              />
              {connectedDevice.batteryLevel !== undefined && (
                <SettingRow
                  icon={<InfoIcon color={colors.primary.DEFAULT} />}
                  title="Battery Level"
                  subtitle={`${connectedDevice.batteryLevel}%`}
                />
              )}
              <View style={styles.deviceButtons}>
                <Button
                  title="Change Device"
                  variant="outline"
                  onPress={() => setDeviceSelectorVisible(true)}
                  style={{ flex: 1 }}
                />
                <Button
                  title="Disconnect"
                  variant="outline"
                  onPress={handleDisconnectDevice}
                  style={{ flex: 1 }}
                />
              </View>
            </>
          ) : (
            <SettingRow
              icon={<BluetoothIcon color={colors.primary.DEFAULT} />}
              title="Add Device"
              subtitle="200+ supported devices from Lovense, Kiiroo, We-Vibe, and more"
              onPress={() => setDeviceSelectorVisible(true)}
            />
          )}
        </Card>

        {/* About Section */}
        <Card style={styles.section}>
          <Text style={styles.sectionTitle}>About</Text>

          <SettingRow
            icon={<InfoIcon color={colors.primary.DEFAULT} />}
            title="FunGen Mobile"
            subtitle="Version 1.0.0"
          />

          <SettingRow
            icon={<InfoIcon color={colors.primary.DEFAULT} />}
            title="Privacy Policy"
            onPress={() => Alert.alert('Privacy Policy', 'Privacy policy will be available soon.')}
          />

          <SettingRow
            icon={<InfoIcon color={colors.primary.DEFAULT} />}
            title="Terms of Service"
            onPress={() => Alert.alert('Terms of Service', 'Terms of service will be available soon.')}
          />
        </Card>
      </ScrollView>

      <DeviceSelector
        visible={deviceSelectorVisible}
        onClose={() => setDeviceSelectorVisible(false)}
        onDeviceConnected={handleDeviceConnected}
        connectedDevice={connectedDevice}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.base,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
    gap: 16,
  },
  section: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    marginBottom: 12,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.DEFAULT,
  },
  settingIcon: {
    width: 40,
    height: 40,
    borderRadius: 8,
    backgroundColor: colors.primary.dim,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  settingContent: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    color: colors.text.primary,
    fontWeight: '500',
  },
  settingSubtitle: {
    fontSize: 14,
    color: colors.text.muted,
    marginTop: 2,
  },
  logoutButton: {
    marginTop: 16,
  },
  authButtons: {
    gap: 12,
  },
  connectedBadge: {
    backgroundColor: colors.status.success + '20',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  connectedBadgeText: {
    color: colors.status.success,
    fontSize: 12,
    fontWeight: '600',
  },
  deviceButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
});
