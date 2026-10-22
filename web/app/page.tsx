'use client';

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { Loader2 } from 'lucide-react';

export default function HomePage() {
  const router = useRouter();
  const pathname = usePathname();
  const { user, loading } = useAuth();

  useEffect(() => {
    if (!loading) {
      if (user) {
        if (pathname === '/' || pathname === '/login' || pathname === '/register') {
          router.push('/chat');
        }
      } else {
        if (pathname !== '/login' && pathname !== '/register') {
          router.push('/login');
        }
      }
    }
  }, [user, loading, router, pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-core-bg">
      <Loader2 className="w-8 h-8 text-electric-blue animate-spin" />
    </div>
  );
}