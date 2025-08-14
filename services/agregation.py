from collections import Counter
from datetime import datetime

from api.dtos import DailySummary, AggregatedReport
from models.monitoring_model import MonitoringSession


def build_daily_summary(sessions: list[MonitoringSession], date: datetime) -> DailySummary:
    if not sessions:
        return DailySummary(
            data=date.strftime("%Y-%m-%d"),
            total_sessoes=0,
            total_tosse=0,
            total_espirro=0,
            total_outros_eventos=0,
            ambiente_predominante="Nenhum",
            duracao_total_minutos=0
        )

    # Contadores
    total_tosse = 0
    total_espirro = 0
    total_outros = 0
    total_duracao = 0
    ambientes = Counter()

    for session in sessions:
        # Contagem de eventos
        events = Counter(e.event_type for e in session.events)
        total_tosse += events.get('Tosse', 0)
        total_espirro += events.get('Espirro', 0)
        total_outros += sum(v for k, v in events.items() if k not in ['Tosse', 'Espirro'])

        # Duração
        if session.start_time and session.end_time:
            total_duracao += int((session.end_time - session.start_time).total_seconds() / 60)

        # Ambientes
        if session.noise_profile:
            ambientes[session.noise_profile] += 1

    # Ambiente predominante (com desempate por ordem de preferência)
    predominant_environment = "Nenhum"
    if ambientes:
        max_count = max(ambientes.values())
        candidates = [env for env, count in ambientes.items() if count == max_count]

        # Ordem de desempate: Silencioso > Moderado > Ruidoso > Outros
        priority_order = ['Ruidoso', 'Moderado', 'Silencioso']
        for env in priority_order:
            if env in candidates:
                predominant_environment = env
                break
        else:
            predominant_environment = candidates[0]  # Pega o primeiro se não estiver na lista de prioridades

    return DailySummary(
        data=date.strftime("%Y-%m-%d"),
        total_sessoes=len(sessions),
        total_tosse=total_tosse,
        total_espirro=total_espirro,
        total_outros_eventos=total_outros,
        ambiente_predominante=predominant_environment,
        duracao_total_minutos=total_duracao
    )

    def build_aggregated_report(sessions, start_date, end_date) -> AggregatedReport:
        """Função auxiliar para construir a resposta do relatório agregado."""

    report_items = monitoring_service.aggregate_sessions_data(sessions)

    total_tosse = sum(item['quantidade_tosse'] for item in report_items)
    total_espirro = sum(item['quantidade_espirro'] for item in report_items)
    total_outros = sum(item['outros_eventos'] for item in report_items)

    return AggregatedReport(
        periodo_inicio=start_date.strftime('%Y-%m-%d'),
        periodo_fim=end_date.strftime('%Y-%m-%d'),
        total_sessoes=len(sessions),
        total_tosse=total_tosse,
        total_espirro=total_espirro,
        total_outros_eventos=total_outros,
        sessoes=report_items
    )
