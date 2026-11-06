import React from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { to: '/', label: '工作台', icon: '▦' },
  { to: '/data', label: '数据采集', icon: '◉' },
  { to: '/models', label: '模型管理', icon: '■' },
  { to: '/training', label: '训练模块', icon: '☆' },
  { to: '/inference', label: '推理模块', icon: '◎' },
  { to: '/inference-history', label: '推理历史', icon: '⧖' },
  { to: '/statistics', label: '统计分析', icon: '≡' },
];

const Sidebar: React.FC = () => {
  const [version, setVersion] = React.useState('...');

  React.useEffect(() => {
    try {
      const v = window.electronAPI?.getElectronVersion?.();
      if (v) setVersion(`v${v}`);
    } catch {
      setVersion('v1.0.0');
    }
  }, []);
  return (
    <aside
      className="glass flex flex-col select-none flex-shrink-0"
      style={{
        width: '240px',
        minWidth: '240px',
        background: 'rgba(247,248,251,0.6)',
        backdropFilter: 'blur(20px)',
        borderRight: '1px solid var(--border)',
      }}
    >
      {/* Logo — match titlebar height */}
      <div
        className="flex items-center px-4 flex-shrink-0"
        style={{ height: '44px', minHeight: '44px' }}
      >
        <span className="text-xs font-bold tracking-wide" style={{ color: 'var(--foreground)' }}>
          Spacecraft
        </span>
        <span className="text-xs ml-2" style={{ color: 'var(--foreground-dim)', fontSize: '11px' }}>
          推理训练平台
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-0.5 px-3 flex-1 py-3 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className="flex items-center gap-3 relative flex-shrink-0"
            style={({ isActive }) => ({
              padding: '10px 12px',
              borderRadius: '10px',
              background: isActive ? 'var(--surface)' : 'transparent',
              border: isActive ? '1px solid var(--border-light)' : '1px solid transparent',
              color: isActive ? 'var(--foreground)' : 'var(--foreground-muted)',
              textDecoration: 'none',
              fontSize: '13px',
              fontWeight: isActive ? 500 : 400,
              transition: 'all var(--transition-fast)',
            })}
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <div
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: '25%',
                      height: '50%',
                      width: '3px',
                      backgroundColor: 'var(--accent)',
                      borderRadius: '0 3px 3px 0',
                    }}
                  />
                )}
                <span className="text-base flex-shrink-0 w-5 text-center">{item.icon}</span>
                <span className="truncate">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Version */}
      <div className="px-4 py-3 flex-shrink-0">
        <span className="text-xs" style={{ color: 'var(--foreground-dim)', fontSize: '11px' }}>
          {version}
        </span>
      </div>
    </aside>
  );
};

export default Sidebar;
