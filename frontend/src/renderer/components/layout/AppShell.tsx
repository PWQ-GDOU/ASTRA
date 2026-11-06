import React from 'react';
import TitleBar from './TitleBar';
import Sidebar from './Sidebar';

interface AppShellProps {
  children: React.ReactNode;
}

const AppShell: React.FC<AppShellProps> = ({ children }) => {
  return (
    <div
      className="flex flex-col"
      style={{
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        backgroundColor: 'var(--bg-base)',
        color: 'var(--foreground)',
        fontFamily: 'var(--font-sans)',
      }}
    >
      <TitleBar />
      <div className="flex flex-row flex-1 overflow-hidden" style={{ minHeight: 0 }}>
        <Sidebar />
        <main
          className="flex-1 overflow-y-auto"
          style={{ padding: '32px 40px', minWidth: 0, minHeight: 0 }}
        >
          {children}
        </main>
      </div>
    </div>
  );
};

export default AppShell;
