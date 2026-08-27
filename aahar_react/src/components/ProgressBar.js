export default function ProgressBar({ label, current, total, unit, color }) {
    const percentage = Math.min((current / total) * 100, 100);

    return (
        <div style={{ marginBottom: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '14px' }}>
                <span style={{ fontWeight: '500' }}>{label}</span>
                <span style={{ color: '#757575' }}>{current} / {total} {unit}</span>
            </div>
            <div style={{
                height: '8px',
                background: '#F5F5F5',
                borderRadius: '4px',
                overflow: 'hidden'
            }}>
                <div style={{
                    height: '100%',
                    width: `${percentage}%`,
                    background: color || 'var(--primary-green)',
                    borderRadius: '4px',
                    transition: 'width 0.5s ease-out'
                }}></div>
            </div>
        </div>
    );
}
